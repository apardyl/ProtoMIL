import argparse
import datetime
import getpass
import glob
import os
import platform
import random
import sys

import matplotlib
import numpy
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from analysis import generate_prototype_activation_matrix
from datasets.breast_dataset import BreastCancerBagsCross
from datasets.colon_dataset import ColonCancerBagsCross
from datasets.mnist_dataset import MnistBags
from datasets.messidor_dataset import DiabeticRetinopathyDataset
from helpers import makedir, str2bool
from model import construct_PPNet
from push import push_prototypes
from save import load_train_state, save_train_state, get_state_path_for_prefix, snapshot_code, \
    load_config_from_train_state, load_model_from_train_state
from settings import COLON_CANCER_SETTINGS, MNIST_SETTINGS, Settings, BREAST_CANCER_SETTINGS, MESSIDOR_SETTINGS
from train_and_test import warm_only, train, joint, test, last_only, TrainMode, valid

matplotlib.use('Agg')

# detect debugger
gettrace = getattr(sys, 'gettrace', None)
DEBUG = gettrace is not None and gettrace()
if DEBUG:
    print('DEBUG MODE - parallelism disabled')

CHECKPOINT_PREFIX = 'checkpoint'
LOGS_DIR = 'runs'
SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FREQUENCY_STEPS = 3

# noinspection PyTypeChecker
parser = argparse.ArgumentParser(prog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpuid', type=int, default=0, help='CUDA device id to use')
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['mnist', 'colon_cancer', 'breast_cancer', 'messidor'],
                    help='Select dataset')
parser.add_argument('-n', '--new_experiment', default=False, action='store_true',
                    help='Overwrite any saved state and start a new experiment (saved checkpoint will be lost)')
parser.add_argument('-l', '--load_state', metavar='STATE_FILE', type=str, default=None,
                    help='Continue training from specified state file (saved checkpoint will be lost)')
parser.add_argument('-c', '--run_name_prefix', type=str, default=None, help='Prefix for the experiment name')
parser.add_argument('-a', '--alloc', type=int, default=None)
parser.add_argument('-w', '--weighting_attention', default=False, action='store_true')
parser.add_argument('--deterministic', type=str2bool, default=True, help='Use deterministic mode (slightly slower)')
for param_name, param_type in Settings.as_params():
    parser.add_argument('--{}'.format(param_name), type=param_type)
args = parser.parse_args()

if args.new_experiment and args.load_state:
    print('You cannot load state and start a new experiment at the same time')
    exit(-1)

checkpoint_file_prefix = '{}.{}.{}'.format(CHECKPOINT_PREFIX, args.dataset, getpass.getuser())
checkpoint_files = get_state_path_for_prefix(checkpoint_file_prefix)
load_state_path = None
if args.load_state:
    load_state_path = args.load_state
elif not args.new_experiment and len(checkpoint_files) > 0:
    if len(checkpoint_files) > 1:
        print('Multiple checkpoint files detected: {}'.format(checkpoint_files))
        print('Leave only one and remove all others to continue or specify which one to use with --load_state')
        exit(1)
    load_state_path = checkpoint_files[0]

config = None
if load_state_path:
    print('Loading state from: {}'.format(load_state_path))
    if args.deterministic:
        print('WARNING: resuming from saved state is not fully deterministic!')
    config = load_config_from_train_state(load_state_path)
if config is None:
    print('Using default base settings for', args.dataset)
    config = {
        'colon_cancer': COLON_CANCER_SETTINGS,
        'breast_cancer': BREAST_CANCER_SETTINGS,
        'mnist': MNIST_SETTINGS,
        'messidor': MESSIDOR_SETTINGS
    }[args.dataset]

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
# print('CUDA available:', torch.cuda.is_available())

if args.alloc:
    mem_holder = torch.randint(0, 1, size=(args.alloc * 1024 * 1024 // 8,), dtype=torch.int64,
                               device=torch.device('cuda'))

config = config.new_from_params(args)
print(config)

if config.random_seed_value is not None:
    seed = config.random_seed_value
elif config.random_seed_id is not None and config.random_seed_id >= 0:
    seed = config.random_seed_presets[config.random_seed_id]
else:
    seed = torch.seed()
torch.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
print('Seed: {}'.format(seed))

if args.deterministic:
    # Make deterministic.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_deterministic(True)
    print('Deterministic mode enabled.')
else:
    print('WARNING: proceeding with non-deterministic mode.')

if args.dataset == 'colon_cancer':
    ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, shuffle_bag=True, data_augmentation=True,
                              fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, push=True, shuffle_bag=True,
                                   fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_valid = ColonCancerBagsCross(path="data/ColonCancer", train=False, all_labels=True, fold_id=config.fold_id,
                                    folds=config.folds, random_state=seed)
    ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, test=True, all_labels=True,
                                   fold_id=config.fold_id, folds=config.folds, random_state=seed)
elif args.dataset == 'breast_cancer':
    ds = BreastCancerBagsCross(path="data/Bisque", train=True, shuffle_bag=True, data_augmentation=True,
                               fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_push = BreastCancerBagsCross(path="data/Bisque", train=True, push=True, shuffle_bag=True,
                                    fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_valid = BreastCancerBagsCross(path="data/Bisque", train=False, all_labels=True, fold_id=config.fold_id,
                                     folds=config.folds, random_state=seed)
    ds_test = BreastCancerBagsCross(path="data/Bisque", train=False, test=True, all_labels=True,
                                    fold_id=config.fold_id, folds=config.folds, random_state=seed)
elif args.dataset == 'messidor':
    path_to_patch_csv = '/shared/sets/datasets/vision/messidor/retina_patches/patches.csv'
    path_to_train_labels_csv = '/shared/sets/datasets/vision/messidor/messidor_scaled_700x700/trainLabels.csv'
    ds = DiabeticRetinopathyDataset(path_to_patch_csv, path_to_train_labels_csv, train=True, shuffle_bag=True, data_augmentation=True, fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_push = DiabeticRetinopathyDataset(path_to_patch_csv, path_to_train_labels_csv, train=True, push=True, shuffle_bag=True, fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_valid = DiabeticRetinopathyDataset(path_to_patch_csv, path_to_train_labels_csv, train=False, fold_id=config.fold_id, folds=config.folds, random_state=seed)
    ds_test = DiabeticRetinopathyDataset(path_to_patch_csv, path_to_train_labels_csv, train=False, test=True, fold_id=config.fold_id, folds=config.folds, random_state=seed)

elif args.dataset == 'mnist':
    ds = MnistBags(train=True, random_state=seed, **config.dataset_settings)
    ds_push = MnistBags(train=True, push=True, random_state=seed, **config.dataset_settings)
    ds_valid = MnistBags(train=False, random_state=seed, **config.dataset_settings, all_labels=True)
    ds_test = MnistBags(train=False, test=True, random_state=seed, **config.dataset_settings, all_labels=True)
else:
    raise NotImplementedError()

print('Dataset loaded.')
print('training set size: {}, push set size: {}, valid set size: {}, test set size: {}'.format(
    len(ds), len(ds_push), len(ds_valid), len(ds_test)))

ppnet = construct_PPNet(base_architecture=config.base_architecture,
                        pretrained=config.pretrained, img_size=config.img_size,
                        prototype_shape=config.prototype_shape,
                        num_classes=config.num_classes,
                        prototype_activation_function=config.prototype_activation_function,
                        add_on_layers_type=config.add_on_layers_type,
                        batch_norm_features=config.batch_norm_features,
                        mil_pooling=config.mil_pooling)

ppnet = ppnet.cuda()

summary(ppnet, (10, 3, config.img_size, config.img_size), col_names=("input_size", "output_size", "num_params"),
        depth=4)

warm_optimizer_specs = [
    {
        'params': ppnet.features.parameters(),
        'lr': config.joint_optimizer_lrs['features'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': config.warm_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': config.warm_optimizer_lrs['prototype_vectors']
    },
    {
        'params': ppnet.attention_V.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_U.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_weights.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    }
]

warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
warm_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(warm_optimizer, gamma=config.warm_lr_gamma)

joint_optimizer_specs = [
    {
        'params': ppnet.features.parameters(),
        'lr': config.joint_optimizer_lrs['features'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': config.joint_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': config.joint_optimizer_lrs['prototype_vectors']
    }
    # {
    #     'params': ppnet.attention_V.parameters(),
    #     'lr': config.last_layer_optimizer_lr['attention']
    # },
    # {
    #     'params': ppnet.attention_U.parameters(),
    #     'lr': config.last_layer_optimizer_lr['attention']
    # },
    # {
    #     'params': ppnet.attention_weights.parameters(),
    #     'lr': config.last_layer_optimizer_lr['attention']
    # }
]

joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=config.joint_lr_step_size,
                                                     gamma=config.joint_lr_gamma)
last_layer_optimizer_specs = [
    {
        'params': ppnet.last_layer.parameters(),
        'lr': config.last_layer_optimizer_lr['last_layer']
    },
    {
        'params': ppnet.attention_V.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_U.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_weights.parameters(),
        'lr': config.last_layer_optimizer_lr['attention']
    }
]

last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

other_state = {
    'joint_optimizer': joint_optimizer,
    'joint_lr_scheduler': joint_lr_scheduler,
    'warm_optimizer': warm_optimizer,
    'last_layer_optimizer': last_layer_optimizer,
}

if load_state_path:
    (step,
     mode,
     epoch,
     iteration,
     experiment_run_name,
     best_accu,
     current_push_best_accu) = load_train_state(load_state_path, ppnet, other_state)
    print('Resuming experiment: {}'.format(experiment_run_name))
    print('Best score so far: {}'.format(best_accu))
else:
    step = 0
    mode = TrainMode.WARM if config.num_warm_epochs > 0 else TrainMode.JOINT
    epoch = 0
    iteration = None
    run_name_prefix = ''
    if args.run_name_prefix:
        run_name_prefix = args.run_name_prefix.lower().replace(' ', '_') + '.'
    time = datetime.datetime.now()
    time = time.replace(microsecond=0)
    experiment_run_name = run_name_prefix + '{}.{}.{}'.format(args.dataset, platform.node(), time.isoformat())
    best_accu = 0.
    current_push_best_accu = 0.
    print('Starting new experiment: {}'.format(experiment_run_name))
    print('Saving code snapshot with git-experiments')
    snapshot_code(experiment_run_name)

model_dir = os.path.join(SAVED_MODELS_PATH, experiment_run_name)
makedir(model_dir)
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)

log_writer = SummaryWriter(os.path.join(LOGS_DIR, experiment_run_name), purge_step=step + 1)

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

workers = 0 if DEBUG else 8

train_loader = torch.utils.data.DataLoader(
    ds, batch_size=None, shuffle=True,
    num_workers=workers,
    pin_memory=False)
train_push_loader = torch.utils.data.DataLoader(
    ds_push, batch_size=None, shuffle=False,
    num_workers=workers,
    pin_memory=False)
valid_loader = torch.utils.data.DataLoader(
    ds_valid, batch_size=None, shuffle=False,
    num_workers=workers,
    pin_memory=False)
test_loader = torch.utils.data.DataLoader(
    ds_test, batch_size=None, shuffle=False,
    num_workers=workers,
    pin_memory=False)

# noinspection PyTypeChecker
log_writer.add_text('dataset_stats',
                    'training set size: {}, push set size: {}, valid set size: {}, test set size: {}'.format(
                        len(train_loader.dataset), len(train_push_loader.dataset), len(valid_loader.dataset),
                        len(test_loader.dataset)),
                    global_step=step)
log_writer.add_text('seed', str(seed), global_step=step)
config_md = '\n'.join(
    ('* ' + x) if idx != 0 else x + '\n' for idx, x in enumerate(str(config).splitlines(keepends=False)))
log_writer.add_text('settings', config_md, global_step=step)
if step == 0:
    log_writer.add_graph(ppnet, next(iter(train_loader))[0].cuda())

# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)


# train the model
print('Training started')


def write_mode(mode: TrainMode, log_writer: SummaryWriter, step: int):
    log_writer.add_scalar('mode/warm', int(mode == TrainMode.WARM), global_step=step)
    log_writer.add_scalar('mode/joint', int(mode == TrainMode.JOINT), global_step=step)
    log_writer.add_scalar('mode/push', int(mode == TrainMode.PUSH), global_step=step)
    log_writer.add_scalar('mode/last_only', int(mode == TrainMode.LAST_ONLY), global_step=step)


last_checkpoint = step
push_model_state_epoch = None

# if epoch < config.push_start and ppnet.mil_pooling == 'gated_attention':
#     ppnet.mil_pooling = 'average'
#     print('\tattention disabled')

# training loop as a state machine.
while True:
    print('step: {}, mode: {}, epoch: {}, iteration: {}'.format(step, mode.name, epoch, iteration))
    if mode == TrainMode.WARM:
        write_mode(TrainMode.WARM, log_writer, step)
        warm_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer, config=config, log_writer=log_writer,
              step=step, weighting_attention=args.weighting_attention)
        warm_lr_scheduler.step()
        accu = valid(model=ppnet, dataloader=valid_loader, config=config, log_writer=log_writer, step=step,
                     weighting_attention=args.weighting_attention)

        test(model=ppnet, dataloader=test_loader, config=config, log_writer=log_writer, step=step,
            weighting_attention=args.weighting_attention)

        push_model_state_epoch = None
        epoch += 1
        if epoch >= config.push_start and epoch in config.push_epochs:
            mode = TrainMode.PUSH
        elif epoch >= config.num_warm_epochs:
            mode = TrainMode.JOINT
    elif mode == TrainMode.JOINT:
        write_mode(TrainMode.JOINT, log_writer, step)
        joint(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer, config=config, log_writer=log_writer,
              step=step, weighting_attention=args.weighting_attention)
        joint_lr_scheduler.step()
        accu = valid(model=ppnet, dataloader=valid_loader, config=config, log_writer=log_writer, step=step,
                     weighting_attention=args.weighting_attention)

        test(model=ppnet, dataloader=test_loader, config=config, log_writer=log_writer, step=step,
            weighting_attention=args.weighting_attention)

        push_model_state_epoch = None
        if epoch >= config.push_start and epoch in config.push_epochs:
            mode = TrainMode.PUSH
        else:
            epoch += 1
    elif mode == TrainMode.PUSH:
        write_mode(TrainMode.PUSH, log_writer, step)
        push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet,
            class_specific=config.class_specific,
            # preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,
            epoch_number=epoch,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True)
        accu = valid(model=ppnet, dataloader=valid_loader, config=config, log_writer=log_writer, step=step,
                     weighting_attention=args.weighting_attention)

        test(model=ppnet, dataloader=test_loader, config=config, log_writer=log_writer, step=step,
            weighting_attention=args.weighting_attention)

        push_model_state_epoch = epoch
        current_push_best_accu = 0.
        if config.mil_pooling == 'gated_attention' and not ppnet.mil_pooling == 'gated_attention':
            ppnet.mil_pooling = 'gated_attention'
            print('\tattention enabled')
        if config.prototype_activation_function != 'linear':
            mode = TrainMode.LAST_ONLY
            iteration = 0
        else:
            mode = TrainMode.JOINT
            epoch += 1
    elif mode == TrainMode.LAST_ONLY:
        write_mode(TrainMode.LAST_ONLY, log_writer, step)
        last_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer, config=config,
              log_writer=log_writer, step=step, weighting_attention=args.weighting_attention)
        accu = valid(model=ppnet, dataloader=valid_loader, config=config, log_writer=log_writer, step=step,
                     weighting_attention=args.weighting_attention)

        test(model=ppnet, dataloader=test_loader, config=config, log_writer=log_writer, step=step,
            weighting_attention=args.weighting_attention)

        iteration += 1
        push_model_state_epoch = epoch
        if iteration >= config.num_last_layer_iterations:
            log_writer.add_figure('prototype_analysis/positive',
                                  generate_prototype_activation_matrix(ppnet, valid_loader, train_push_loader, epoch,
                                                                       model_dir, torch.device('cuda'), bag_class=1)
                                  , global_step=step)
            log_writer.add_figure('prototype_analysis/negative',
                                  generate_prototype_activation_matrix(ppnet, valid_loader, train_push_loader, epoch,
                                                                       model_dir, torch.device('cuda'), bag_class=0)
                                  , global_step=step)
            iteration = None
            epoch += 1
            mode = TrainMode.JOINT
    else:
        raise NotImplementedError('unknown mode')
    if epoch >= config.num_train_epochs:
        break
    step += 1

    do_snapshot = False

    if last_checkpoint + CHECKPOINT_FREQUENCY_STEPS <= step:
        do_snapshot = True

    if accu > best_accu:
        best_accu = accu
        do_snapshot = True
        best_model_path = os.path.join(model_dir, 'best')
        save_train_state(best_model_path, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu, accu, config)
        print('New best score: {}, saving snapshot'.format(accu))

    if push_model_state_epoch is not None and accu > current_push_best_accu:
        current_push_best_accu = accu
        do_snapshot = True
        push_best_model_path = os.path.join(model_dir, '{}.push.best'.format(push_model_state_epoch))
        save_train_state(push_best_model_path, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu, accu, config)
        print('Push {} best score: {}, saving snapshot'.format(epoch, accu))

    if do_snapshot:
        print('Saving checkpoint')
        save_train_state(checkpoint_file_prefix, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu, accu, config)
        last_checkpoint = step

# test
# find push with max accuracy
score_model_with_max_push_acc = -1
path_to_model_with_max_push_acc = None
for i in config.push_epochs:
    if i >= config.push_start and i < config.num_train_epochs:
        model_push_path = glob.glob(model_dir + f"/{i}*")[0]
        score = float(".".join(model_push_path.split(".")[-3:-1]))
        if score >= score_model_with_max_push_acc:
            score_model_with_max_push_acc = score
            path_to_model_with_max_push_acc = model_push_path

ppnet_test = construct_PPNet(base_architecture=config.base_architecture,
                             pretrained=config.pretrained,
                             img_size=config.img_size,
                             prototype_shape=config.prototype_shape,
                             num_classes=config.num_classes,
                             prototype_activation_function=config.prototype_activation_function,
                             add_on_layers_type=config.add_on_layers_type,
                             batch_norm_features=config.batch_norm_features)

print('load model from ' + path_to_model_with_max_push_acc)
load_model_from_train_state(path_to_model_with_max_push_acc, ppnet_test)

ppnet_test = ppnet_test.cuda()

accu = test(model=ppnet_test, dataloader=test_loader, config=config, log_writer=log_writer, step=step+1,
            weighting_attention=args.weighting_attention)

epoch_for_ppnet_test = path_to_model_with_max_push_acc.split(".")[-7].split("/")[-1]

log_writer.add_figure('test_prototype_analysis/positive',
                      generate_prototype_activation_matrix(ppnet_test, test_loader, train_push_loader,
                                                           epoch_for_ppnet_test,
                                                           model_dir, torch.device('cuda'), bag_class=1)
                      , global_step=step)
log_writer.add_figure('test_prototype_analysis/negative',
                      generate_prototype_activation_matrix(ppnet_test, test_loader, train_push_loader,
                                                           epoch_for_ppnet_test,
                                                           model_dir, torch.device('cuda'), bag_class=0)
                      , global_step=step)

best_model_path = os.path.join(model_dir, 'end')
save_train_state(best_model_path, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                 best_accu, current_push_best_accu, accu, config)
[os.remove(checkpoint) for checkpoint in get_state_path_for_prefix(checkpoint_file_prefix)]
log_writer.flush()
log_writer.close()

if args.alloc:
    print(mem_holder[0].item())
