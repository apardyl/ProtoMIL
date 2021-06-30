from os import replace
from skimage import io, color 
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import utils_augemntation
from typing import List
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

normalization_mean = [89.7121552586411, 89.7121552586411, 89.7121552586411]
normalization_std = [18.49568745464706, 15.415668522447366, 11.147622232506315]


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, patches_file, label_file, train=True, test=False, shuffle_bag=False,
                 data_augmentation=False, push=False,
                 folds=10, fold_id=1, random_state=3):
        self.train = train
        self.test = test
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.push = push
        self.folds = folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.df_patches = pd.read_csv(patches_file)
        self.df_labels = pd.read_csv(label_file, sep=';')
        self.df_labels = self.df_labels[self.df_labels['image'].apply(lambda x: x.split('.')[0]).isin(self.df_patches.image.values)]
        self.list_image_names = list(self.df_labels['image'].apply(lambda x: x.split('.')[0]))
        
        self.r = np.random.RandomState(random_state)
        
        tr = [ToTensor(),
                Normalize(mean=normalization_mean, std=normalization_std)
              ]
        tst = [ToTensor(),
                Normalize(mean=normalization_mean, std=normalization_std)
               ]

        psh = [ToTensor()]
        
        self.data_augmentation_img_transform = Compose(tr)

        self.normalize_to_tensor_transform = Compose(tst)

        self.to_tensor_transform = Compose(psh)

        folds = list(KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state).split(self.list_image_names))
        
        if self.test:
            indices = set(folds[self.fold_id][1])
        else:
            if self.train:
                val_indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]), replace=False)
                indices = set(folds[self.fold_id][0]) - set(val_indices)
            else:  # valid
                indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]), replace=False)

        self.bags, self.labels = self.create_bags(np.asarray(self.list_image_names)[list(indices)])

    def transform_and_data_augmentation(self, bag, raw=False):
        if raw:
            img_transform = self.to_tensor_transform
        elif not raw and self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            bag_tensors.append(img_transform(img))

        return torch.stack(bag_tensors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = torch.tensor(self.labels[idx]).sign().unsqueeze(0)

        if self.push:
            return self.transform_and_data_augmentation(bag, raw=True), self.transform_and_data_augmentation(
                bag), label
        else:
            return self.transform_and_data_augmentation(bag), label

    def create_bags(self, list_image_names):
        labels, bags = [], []
        for i, row in self.df_labels.iterrows():
            label = row['level']
            img_id = row['image'].split('.')[0]
            if img_id in list_image_names:
                curr_paths = self.df_patches[self.df_patches['image'] == img_id]['path'].values
                curr_bag = []
                for path in curr_paths:
                    curr_bag.append(io.imread(path))
                labels.append(label)
                bags.append(curr_bag)
        return bags, labels