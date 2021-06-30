import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import find_nearest
from datasets.mnist_dataset import MnistBags
from helpers import makedir
from model import construct_PPNet
from save import load_model_from_train_state
from settings import MNIST_SETTINGS


def generate_prototype_activation_matrix(ppnet, test_dataloader, push_dataloader, epoch,
                                         model_dir, device, bag_class=0, N=10):
    print('    analysis for class', bag_class)
    epoch_number_str = str(epoch)
    load_img_dir = os.path.join(model_dir, 'img')

    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-' + epoch_number_str, 'bb' + epoch_number_str + '.npy'))
    prototype_img_identity = prototype_info[:, -1]
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    # print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    # print('Their class identities are: ' + str(prototype_img_identity))

    bag, label = next(((b, l) for b, l in iter(test_dataloader) if l.max().unsqueeze(0) == bag_class and len(b) >= N))
    
    count_positive_patches = sum(label)
    if len(label) > 1:
        label = label.max().unsqueeze(0)
    
    bag = bag.squeeze(0)

    with torch.no_grad():
        ppnet.eval()

        images_test = bag.to(device)
        labels_test = label.to(device)

        logits, min_distances, attention, vector_scores = ppnet.forward_(
            images_test)  # function forward in model.py should return logits, min_distances, A, prototype_activations

        conv_output, distances = ppnet.push_forward(images_test)
        prototype_activation_patterns = ppnet.distance_2_similarity(distances)
        if ppnet.prototype_activation_function == 'linear':
            prototype_activation_patterns = prototype_activation_patterns + max_dist

        tables = []
        for i in range(logits.size(0)):
            tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
            # print(str(i) + ' ' + str(tables[-1]))

        idx = 0
        predicted_cls = tables[idx][0]
        correct_cls = tables[idx][1]
        # print('Predicted: ' + str(predicted_cls))
        # print('Actual: ' + str(correct_cls))

    # Take the N patches with the most attention
    at = attention.squeeze(0).detach().cpu().numpy()
    top_patches = at.argsort()[-N:][::-1]
    # print(f'        patch indexes: {top_patches}')

    imgs = [bag[i].permute(1, 2, 0) for i in top_patches]

    # Take the most highly activated area of the image by prototype
    imgs_with_self_activation_by_prototype = []
    for img, idx in zip(imgs, top_patches):
        original_img = img

        self_activation_for_img = []
        # for every prototype
        for i in range(len(prototype_img_identity)):
            activation_pattern = prototype_activation_patterns[idx][i].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(original_img.shape[0], original_img.shape[1]),
                                                    interpolation=cv2.INTER_CUBIC)

            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap

            self_activation_for_img.append(np.asarray(overlayed_img))

        imgs_with_self_activation_by_prototype.append(np.asarray(self_activation_for_img))

    ### Take prototypes

    prototype_dir = os.path.join(load_img_dir, 'epoch-' + epoch_number_str)

    prototypes = []
    for i in range(len(prototype_img_identity)):
        prototypes.append(plt.imread(f'{prototype_dir}/prototype-img{i}.png'))

    # Take prototypes img original with self activation
    prototypes_img_with_act = []
    for i in range(len(prototype_img_identity)):
        prototypes_img_with_act.append(plt.imread(f'{prototype_dir}/prototype-img-original_with_self_act{i}.png'))

    ### Find the k-nearest patches in the dataset to each prototype

    k = 5

    root_dir_for_saving_train_images = os.path.join(model_dir)
    makedir(root_dir_for_saving_train_images)

    find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=push_dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
        ppnet=ppnet,  # pytorch network with prototype_vectors
        k=k,
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

    k_nearest_patches = []

    for i in range(len(prototype_img_identity)):
        tmp = []
        for j in range(1, k + 1):
            tmp.append(plt.imread(f'{root_dir_for_saving_train_images}/{i}/nearest-{j}_original_with_heatmap.png'))
        k_nearest_patches.append(np.asarray(tmp))

    ###  Vector score for top patches
    arr = vector_scores.detach().cpu().numpy()
    arr = np.array([arr[i] for i in top_patches])

    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    grid_score = np.around(arr.T, 2)
    colors = get_colors(grid_score, plt.cm.magma)
    len_proto = len(prototype_img_identity)

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(2 * N + 2 + k, len_proto + 2))
    fig.suptitle(f'patches in bag: {len(bag)}, positive patches: {count_positive_patches}, class label: {label.item()}', fontsize=40)

    grid = plt.GridSpec(len_proto + 2, 2 * N + 2 + k, hspace=0.04, wspace=0.04)

    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    # histogram
    l = 0
    for j in range(3 + k, 2 * N + 3 + k, 2):
        for i in range(2, len_proto + 2):
            main_ax = fig.add_subplot(grid[i, j])
            main_ax.set_facecolor(colors[i - 2][l])
            main_ax.text(0.5 * (left + right), 0.5 * (bottom + top), grid_score[i - 2][l],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=15,
                        color='white' if grid_score[i - 2][l] < grid_score.max() * 0.9 else 'black',
                        transform=main_ax.transAxes)

            main_ax.get_xaxis().set_visible(False)
            main_ax.get_yaxis().set_visible(False)
        l = l + 1

    # images with self activation by prototype
    l = 0
    for j in range(2 + k, 2 * N + 2 + k, 2):
        for i in range(2, len_proto + 2):
            main_ax = fig.add_subplot(grid[i, j])
            main_ax.set_xlim([0, 27])
            main_ax.set_ylim([0, 27])
            main_ax.invert_yaxis()
            main_ax.imshow(imgs_with_self_activation_by_prototype[l][i - 2], aspect='auto')

            main_ax.get_xaxis().set_visible(False)
            main_ax.get_yaxis().set_visible(False)
        l = l + 1

    # prototypes
    for i in range(2, len_proto + 2):
        main_ax = fig.add_subplot(grid[i, 1 + k])
        main_ax.invert_yaxis()
        main_ax.imshow(prototypes[i - 2])

        main_ax.get_xaxis().set_visible(False)
        main_ax.get_yaxis().set_visible(False)

    # prototypes images with activation
    for i in range(2, len_proto + 2):
        main_ax = fig.add_subplot(grid[i, 0 + k])
        main_ax.set_xlim([0, 27])
        main_ax.set_ylim([0, 27])
        main_ax.invert_yaxis()
        main_ax.imshow(prototypes_img_with_act[i - 2])

        if i - 2 < len_proto // 2:
            main_ax.patch.set_edgecolor('red')
        else:
            main_ax.patch.set_edgecolor('green')
        main_ax.patch.set_linewidth('5')

        main_ax.get_xaxis().set_visible(False)
        main_ax.get_yaxis().set_visible(False)

    # k nearest patches
    for j in range(k):
        for i in range(2, len_proto + 2):
            main_ax = fig.add_subplot(grid[i, j])
            main_ax.set_xlim([0, 27])
            main_ax.set_ylim([0, 27])
            main_ax.invert_yaxis()
            main_ax.imshow(k_nearest_patches[i - 2][j])

            main_ax.get_xaxis().set_visible(False)
            main_ax.get_yaxis().set_visible(False)

    # patches
    l = 0
    for i in range(2 + k, 2 * N + 2 + k, 2):
        main_ax = fig.add_subplot(grid[0:2, i:i + 2])
        main_ax.set_xlim([0, 27])
        main_ax.set_ylim([0, 27])
        main_ax.invert_yaxis()
        main_ax.set_title(f'{at[top_patches[l]]:.3f}', fontsize=25)
        main_ax.imshow(imgs[l], aspect='auto')

        main_ax.get_xaxis().set_visible(False)
        main_ax.get_yaxis().set_visible(False)
        l = l + 1

    plt.axis('off')
    return fig


if __name__ == '__main__':
    # %%

    device = torch.device('cuda')

    # %%

    config = MNIST_SETTINGS

    load_model_dir = 'saved_models/mnist.vulcan.2021-03-16T16:42:33.070310'  # 50

    load_model_name = '20.push.best.57.94.00.pck'

    load_model_path = os.path.join(load_model_dir, load_model_name)

    ppnet = construct_PPNet(base_architecture=config.base_architecture,
                            pretrained=False,
                            img_size=config.img_size,
                            prototype_shape=config.prototype_shape,
                            num_classes=config.num_classes,
                            prototype_activation_function=config.prototype_activation_function,
                            add_on_layers_type=config.add_on_layers_type,
                            batch_norm_features=config.batch_norm_features)
    ppnet = ppnet.to(device)

    print('load model from ' + load_model_path)
    load_model_from_train_state(load_model_path, ppnet)

    ds_test = MnistBags(train=False, bag_length_mean=50, bag_length_std=2, positive_samples_in_bag_ratio_mean=0.1,
                        positive_samples_in_bag_ratio_std=0.02)

    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    ds_push = MnistBags(train=True, push=True)

    push_loader = torch.utils.data.DataLoader(
        ds_push, batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    fig = generate_prototype_activation_matrix(ppnet, test_loader, push_loader, 20, load_model_dir, device, bag_class=1)
    plt.show()
