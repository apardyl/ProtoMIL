from enum import Enum

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, \
    average_precision_score
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as func

from focalloss import FocalLoss
from helpers import list_of_distances
from settings import Settings
from weight_loss import WeightCrossEntropyLoss


class TrainMode(Enum):
    WARM = 'warm'
    JOINT = 'joint'
    PUSH = 'push'
    LAST_ONLY = 'last_only'


def _train_or_test(model, dataloader, config: Settings, optimizer=None, use_l1_mask=True,
                   log_writer: SummaryWriter = None, step: int = 0, multilabel=False):
                   #bces=[], cl_csts=[], sep_csts=[]):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    batch_size = 1  # 16
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_loss = 0
    if multilabel:
        conf_matrix = np.zeros((20, 2, 2), dtype='int32')
    else:
        conf_matrix = np.zeros((2, 2), dtype='int32')
    preds = []
    targets = []

    if config.loss_function == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif config.loss_function == 'focal':
        loss_fn = FocalLoss(alpha=0.5, gamma=2)
    elif config.loss_function == 'binary_cross_entropy':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise NotImplementedError('unknown loss function: ' + config.loss_function)

    if is_train:
        optimizer.zero_grad()

    n_bags = len(dataloader)
    n_batches = n_bags // batch_size
    batches = [batch_size for i in range(n_batches)]
    if n_bags % batch_size:
        batches.append(n_bags % batch_size)
        n_batches += 1

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        current_batch_size = batches[i // batch_size]

        if not multilabel and len(label) > 1:
            label = label.max()

        label = label.unsqueeze(0)
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances, attention, _ = model.forward_(input)

            if multilabel:
                bce_vals = loss_fn(output, target)
                n_positive = torch.count_nonzero(label)
                n_negative = label.shape[1] - n_positive
                w = n_negative / n_positive
                bce_vals[label == 1] *= w
                cross_entropy = torch.mean(bce_vals)
            else:
                cross_entropy = loss_fn(output, target)
            if config.mil_pooling == 'loss_attention':
                instance_labels = target * torch.ones(input.size(0), dtype=torch.long, device=input.device)
                loss_2 = WeightCrossEntropyLoss()(model.out_c, instance_labels, model.A)
                cross_entropy += 2.0 * loss_2

            if config.class_specific:
                max_dist = (model.prototype_shape[1]
                            * model.prototype_shape[2]
                            * model.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                
                # a = attention.detach().cpu()
                # tmp = np.interp(a, (a.min(), a.max()), (0.001, 1))
                # m = torch.tensor(tmp).cuda()

                # calculate cluster cost
                if multilabel:
                    class_idx = []
                    for idx, l in enumerate(label.squeeze()):
                        if l == 1:
                            class_idx.append(idx)
                else:
                    class_idx = label

                prototype_identity = model.prototype_class_identity[:, class_idx]
                if len(class_idx) > 1:
                    prototype_identity = torch.sum(prototype_identity, dim=1, keepdim=True)
                prototypes_of_correct_class = torch.t(prototype_identity).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if multilabel:
                    l1 = 0
                    for c in range(model.num_classes):
                        if use_l1_mask:
                            l1_mask = 1 - torch.t(model.prototype_class_identity_2[c]).cuda()
                            l1_class = (model.last_layer_heads[c].weight * l1_mask).norm(p=1)
                        else:
                            l1_class = model.last_layer_heads[c].weight.norm(p=1)
                    l1 += l1_class
                    l1 /= model.num_classes

                else:
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                        l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            targets.append(target.cpu().numpy())

            if multilabel:
                probs = torch.sigmoid(output.data)
                predicted = torch.where(probs >= 0.5, 1, 0).cpu().numpy()  # todo: select threshold
                # predicted = torch.round(probs).cpu().numpy()
                target = target.cpu().numpy()
                preds.append(probs.detach().cpu())
                conf_matrix += multilabel_confusion_matrix(target, predicted)
            else:
                pred_s = func.softmax(output, dim=-1)
                preds.append(pred_s.data.cpu().numpy())
                conf_matrix += confusion_matrix(target.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])

            # n_batches += 1
            total_cross_entropy += cross_entropy.item() / current_batch_size
            total_cluster_cost += cluster_cost.item() / current_batch_size
            total_separation_cost += separation_cost.item() / current_batch_size
            total_avg_separation_cost += avg_separation_cost.item() / current_batch_size

        # bces.append(cross_entropy.item())
        # cl_csts.append(cluster_cost.item())
        # sep_csts.append(separation_cost.item())
        # compute gradient and do SGD step
        if config.class_specific:
            loss = (config.coef_crs_ent * cross_entropy
                    + config.coef_clst * cluster_cost
                    + config.coef_sep * separation_cost
                    + config.coef_l1 * l1)
        else:
            loss = (config.coef_crs_ent * cross_entropy
                    + config.coef_clst * cluster_cost
                    + config.coef_l1 * l1)
        total_loss += loss.item() / current_batch_size
        if is_train:
            loss /= current_batch_size
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping
            if (i+1) % batch_size == 0 or (i+1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        del input
        del target
        del output
        del predicted
        del min_distances

    total_cross_entropy /= n_batches
    total_cluster_cost /= n_batches
    total_separation_cost /= n_batches
    total_loss /= n_batches
    total_avg_separation_cost /= n_batches

    # print(" -------- BCE:", np.mean(bces))
    # print(" ---- Cluster:", np.mean(cl_csts))
    # print(" -------- Sep:", np.mean(sep_csts))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if not multilabel:
        auc = roc_auc_score(targets, preds[..., 1])

    if multilabel:
        print('\t\tAP:', average_precision_score(targets, preds, average=None))
        print('\t\tmAP:', average_precision_score(targets, preds))
    else:
        print('\t\taccuracy:', n_correct / n_examples)
        print('\t\tauc:', auc)
    print('\t\ttotal_loss:', total_loss)

    suffix = '/train' if is_train else '/test'
    if log_writer:

        log_writer.add_scalar('total_loss' + suffix, total_loss, global_step=step)
        log_writer.add_scalar('cross_entropy' + suffix, total_cross_entropy, global_step=step)
        log_writer.add_scalar('cluster_cost' + suffix, total_cluster_cost, global_step=step)

        if config.class_specific:
            log_writer.add_scalar('separation_cost' + suffix, total_separation_cost, global_step=step)
            log_writer.add_scalar('avg_separation_cost' + suffix, total_avg_separation_cost,
                                  global_step=step)
        if multilabel:
            log_writer.add_scalar('mAP' + suffix, average_precision_score(targets, preds), global_step=step)
            for i in range(model.num_classes):
                conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix[i]).plot(cmap='Blues', values_format='d')
                nr = '_' + str(i+1)
                log_writer.add_figure('confusion_matrix' + nr + suffix, conf_plot.figure_, global_step=step, close=True)
        else:
            log_writer.add_scalar('accuracy' + suffix, n_correct / n_examples, global_step=step)
            log_writer.add_scalar('auc' + suffix, auc, global_step=step)
            conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix[0]).plot(cmap='Blues', values_format='d')
            log_writer.add_figure('confusion_matrix' + suffix, conf_plot.figure_, global_step=step, close=True)

    if multilabel:
        l1 = 0
        for c in range(model.num_classes):
            l1 += model.last_layer_heads[c].weight.norm(p=1).item()
        l1 /= model.num_classes
    log_writer.add_scalar('l1' + suffix, l1, global_step=step)

    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))

    if log_writer:
        log_writer.add_scalar('p_avg_pair_dist' + suffix, p_avg_pair_dist, global_step=step)

    if multilabel:
        return average_precision_score(targets, preds)
    return n_correct / n_examples


def train(model, dataloader, optimizer, config: Settings, log_writer: SummaryWriter = None, step: int = 0, multilabel=False):
    assert (optimizer is not None)

    print('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, config=config, optimizer=optimizer, log_writer=log_writer,
                          step=step, multilabel=multilabel)


def test(model, dataloader, config: Settings, log_writer: SummaryWriter = None, step: int = 0, multilabel=False):
    print('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, config=config, optimizer=None,
                          log_writer=log_writer, step=step, multilabel=multilabel)


def _freeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = False


def _unfreeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = True


def last_only(model):
    _freeze_layer(model.features)
    _freeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = False

    if model.multilabel:
        for attention_head, last_layer_head in zip(model.attention_heads, model.last_layer_heads):
            _freeze_layer(attention_head.attention_V)
            _freeze_layer(attention_head.attention_U)
            _freeze_layer(attention_head.attention_weights)

            _unfreeze_layer(last_layer_head)
    else:
        _freeze_layer(model.attention_V)
        _freeze_layer(model.attention_U)
        _freeze_layer(model.attention_weights)

        _unfreeze_layer(model.last_layer)


def warm_only(model):
    _freeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = True

    if model.multilabel:
        for attention_head, last_layer_head in zip(model.attention_heads, model.last_layer_heads):
            _unfreeze_layer(attention_head.attention_V)
            _unfreeze_layer(attention_head.attention_U)
            _unfreeze_layer(attention_head.attention_weights)

            _freeze_layer(last_layer_head)
    else:
        _unfreeze_layer(model.attention_V)
        _unfreeze_layer(model.attention_U)
        _unfreeze_layer(model.attention_weights)

        _freeze_layer(model.last_layer)


def joint(model):
    _freeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = True

    if model.multilabel:
        for attention_head, last_layer_head in zip(model.attention_heads, model.last_layer_heads):
            _unfreeze_layer(attention_head.attention_V)
            _unfreeze_layer(attention_head.attention_U)
            _unfreeze_layer(attention_head.attention_weights)

            _freeze_layer(last_layer_head)
    else:
        _unfreeze_layer(model.attention_V)
        _unfreeze_layer(model.attention_U)
        _unfreeze_layer(model.attention_weights)

        _freeze_layer(model.last_layer)


# def warm_only(model):
#     _unfreeze_layer(model.features)
#     _unfreeze_layer(model.add_on_layers)
#
#     _unfreeze_layer(model.attention_V)
#     _unfreeze_layer(model.attention_U)
#     _unfreeze_layer(model.attention_weights)
#
#     model.prototype_vectors.requires_grad = True
#
#     _unfreeze_layer(model.last_layer)