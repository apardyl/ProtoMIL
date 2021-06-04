from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize


class PascalBags(Dataset):
    def __init__(self, root='data/PascalVOC', train=True, push=False):
        self.root = root
        if push:
            train = True
        self.dataset = {True: 'train', False: 'test'}[train]
        self.push = push
        self.bags, self.labels = self.create_bags()

    def create_bags(self):
        sets_path = self.root + '/ImageSets/'
        images_path = self.root + '/JPGImages/'
        bboxes_path = self.root + '/BoundingBoxes/'

        classes, class2images = self.__load_labels(sets_path)

        bags, labels = [], []
        images = open(sets_path + self.dataset + '.txt')
        for img_name in images:
            img_name = img_name.strip()
            img = Image.open(images_path + img_name + '.jpg')
            img = np.asarray(img)

            bag = []
            bboxes = open(bboxes_path + img_name + '.txt')
            bboxes = list(bboxes)[1:]
            for line in bboxes:
                conf, min_x, min_y, max_x, max_y = line.strip().split(', ')
                if float(conf) <= -0.97:
                    break
                min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
                patch = img[min_y: max_y, min_x: max_x]
                bag.append(patch)

            # for i, patch in enumerate(bag):
            #     plt.imshow(patch)
            #     plt.show()

            label = torch.zeros(len(classes))
            for i, class_name in enumerate(classes):
                if img_name in class2images[class_name]:
                    label[i] = 1  # omitting 'difficult' examples (with label 0)

            bags.append(bag)
            labels.append(label)

        return bags, labels

    def __load_labels(self, sets_path):
        """Map classes to image names."""

        classes = open(sets_path + 'class.txt')
        classes = [class_name.strip() for class_name in list(classes)]
        class2images = {}
        for class_name in classes:
            class_name = class_name.strip()
            imgs_with_labels = open(sets_path + class_name + '_' + self.dataset + '.txt')
            imgs_with_labels = list(imgs_with_labels)
            images = []
            for line in imgs_with_labels:
                img_name, label = line.split()
                if label == '1':
                    images.append(img_name)
            class2images[class_name] = images

        return classes, class2images

    def __transform(self, bag, normalize=True):
        if normalize:
            transform = Compose([ToTensor(), Resize((224, 224)), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transform = Compose([ToTensor(), Resize((224, 224))])

        bag_tensors = []
        for patch in bag:
            patch = Image.fromarray(patch)
            bag_tensors.append(transform(patch))
        return torch.stack(bag_tensors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        bag = self.bags[item]
        label = self.labels[item]
        if self.push:
            return self.__transform(bag, normalize=False), self.__transform(bag), label
        else:
            return self.__transform(bag), label
