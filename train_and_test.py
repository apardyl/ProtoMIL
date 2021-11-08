from enum import Enum

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score, \
    f1_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as func
from tqdm import tqdm

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
                   log_writer: SummaryWriter = None, step: int = 0, weighting_attention=False, is_valid=False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
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
    conf_matrix = np.zeros((2, 2), dtype='int32')
    preds = []
    targets = []

    if config.loss_function == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif config.loss_function == 'binary_cross_entropy':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config.loss_function == 'focal':
        loss_fn = FocalLoss(alpha=0.5, gamma=2)
    else:
        raise NotImplementedError('unknown loss function: ' + config.loss_function)

    with tqdm(total=len(dataloader.dataset), unit='bag') as pbar:
        for i, (image, label) in enumerate(dataloader):
            input = image.cuda()

            # if param all_labels=True in dataloader, set label to positive if at least one in the list
            if len(label) > 1:
                label = label.max().unsqueeze(0)

            target = label.cuda()

            # torch.enable_grad() has no effect outside of no_grad()
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                output, min_distances, attention, _ = model.forward_(input)

                output = output.squeeze(0)
                target = target.float()
                #print('output ', output)
                #print('target ', target)
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

                    attention_detached = attention.detach().cpu()
                    weight = np.interp(attention_detached, (attention_detached.min(), attention_detached.max()), (0.001, 1))
                    #weight += 1

                    if weighting_attention:
                        tensor_weight = torch.tensor(weight).cuda()
                    else:
                        tensor_weight = torch.tensor(1).cuda()

                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, 0]).cuda()
                    if label == 1:
                        inverted_distances, _ = torch.max(
                            (max_dist - (min_distances * tensor_weight.T)) * prototypes_of_correct_class, dim=1)
                        cluster_cost = torch.mean(max_dist - inverted_distances)
                    else:
                        cluster_cost = torch.tensor(0)

                    # calculate separation cost
                    prototypes_of_wrong_class = torch.t(model.prototype_class_identity[:, 0]).cuda()
                    if label == 0:
                        inverted_distances_to_nontarget_prototypes, _ = \
                            torch.max((max_dist - (min_distances * tensor_weight.T)) * prototypes_of_wrong_class, dim=1)
                        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                    else:
                        separation_cost = torch.tensor(0)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum((min_distances * tensor_weight.T) * prototypes_of_wrong_class, dim=0) / torch.sum(
                            prototypes_of_wrong_class,
                            dim=0)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                        l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.last_layer.weight.norm(p=1)

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.last_layer.weight.norm(p=1)

                # evaluation
                predicted = torch.round(torch.sigmoid(output)).detach()
                #print('predicted ', predicted)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                pred_s = torch.sigmoid(output).detach() # func.softmax(output, dim=-1)
                preds.append(pred_s.data.cpu().numpy())
                targets.append(target.cpu().numpy())

                conf_matrix += confusion_matrix(target.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])

                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

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
            total_loss += loss.item()
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping
                optimizer.step()

            del input
            del target
            del output
            del predicted
            del min_distances
            pbar.update(1)

    total_cross_entropy /= n_batches
    total_cluster_cost /= n_batches
    total_separation_cost /= n_batches
    total_loss /= n_batches

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # print('preds ', preds)
    # print('targets ', targets)
    auc = roc_auc_score(targets, preds)
    pred_y = preds.round()
    # print('pred_y ', pred_y)
    precision = precision_score(targets, pred_y, zero_division=0)
    recall = recall_score(targets, pred_y, zero_division=0)
    f1 = f1_score(targets, pred_y, zero_division=0)
    fpr, tpr, threshold = roc_curve(targets, preds)

    print('\t\taccuracy:', n_correct / n_examples)
    print('\t\tauc:', auc)
    print('\t\ttotal_loss:', total_loss)

    if is_train:
        suffix = '/train'
    else:
        if is_valid:
            suffix = '/valid'
        else:
            suffix = '/test'

    if log_writer:

        log_writer.add_scalar('total_loss' + suffix, total_loss, global_step=step)
        log_writer.add_scalar('cross_entropy' + suffix, total_cross_entropy, global_step=step)
        log_writer.add_scalar('cluster_cost' + suffix, total_cluster_cost, global_step=step)

        if config.class_specific:
            log_writer.add_scalar('separation_cost' + suffix, total_separation_cost, global_step=step)
            log_writer.add_scalar('avg_separation_cost' + suffix, total_avg_separation_cost / n_batches,
                                  global_step=step)

        log_writer.add_scalar('accuracy' + suffix, n_correct / n_examples, global_step=step)
        log_writer.add_scalar('auc' + suffix, auc, global_step=step)
        log_writer.add_scalar('precision' + suffix, precision, global_step=step)
        log_writer.add_scalar('recall' + suffix, recall, global_step=step)
        log_writer.add_scalar('f-score' + suffix, f1, global_step=step)
        log_writer.add_scalar('l1' + suffix, model.last_layer.weight.norm(p=1).item(), global_step=step)
        conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(cmap='Blues', values_format='d')
        log_writer.add_figure('confusion_matrix' + suffix, conf_plot.figure_, global_step=step, close=True)

        plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        log_writer.add_figure('roc' + suffix, plt.gcf(), global_step=step, close=True)


    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))

    if log_writer:
        log_writer.add_scalar('p_avg_pair_dist' + suffix, p_avg_pair_dist, global_step=step)

    return auc


def train(model, dataloader, optimizer, config: Settings, log_writer: SummaryWriter = None,
          step: int = 0, weighting_attention=False):
    assert (optimizer is not None)

    print('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, config=config, optimizer=optimizer,
                          log_writer=log_writer, step=step, weighting_attention=weighting_attention)


def valid(model, dataloader, config: Settings, log_writer: SummaryWriter = None, step: int = 0,
          weighting_attention=False):
    print('\tvalid')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, config=config, optimizer=None,
                          log_writer=log_writer, step=step, weighting_attention=weighting_attention, is_valid=True)


def test(model, dataloader, config: Settings, log_writer: SummaryWriter = None, step: int = 0,
         weighting_attention=False):
    print('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, config=config, optimizer=None,
                          log_writer=log_writer, step=step, weighting_attention=weighting_attention)


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

    _unfreeze_layer(model.attention_V)
    _unfreeze_layer(model.attention_U)
    _unfreeze_layer(model.attention_weights)
    _unfreeze_layer(model.last_layer)


def warm_only(model):
    _unfreeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    _unfreeze_layer(model.attention_V)
    _unfreeze_layer(model.attention_U)
    _unfreeze_layer(model.attention_weights)

    model.prototype_vectors.requires_grad = True

    _unfreeze_layer(model.last_layer)


def joint(model):
    _unfreeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = True

    _freeze_layer(model.attention_V)
    _freeze_layer(model.attention_U)
    _freeze_layer(model.attention_weights)
    _freeze_layer(model.last_layer)
