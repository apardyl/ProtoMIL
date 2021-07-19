from os import replace
from skimage import io, color 
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
from PIL import ImageFile
from torchvision.transforms.transforms import RandomVerticalFlip, ToPILImage
ImageFile.LOAD_TRUNCATED_IMAGES = True
import utils_augemntation
from typing import List
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import utils_augemntation
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from typing import List, Callable, Optional

from torchvision.transforms import ToTensor, Normalize, Compose

normalization_mean = [144.33171767592685/255, 67.48043553767825/255, 22.65431090601474/255]
normalization_std = [26.823169068307216/255, 16.366394611772588/255, 7.449466376062873/255]

TRANSFORMS = Compose([ToTensor(), Normalize(mean=normalization_mean, std=normalization_std)])

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
        
        tr = [
            ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # GaussianNoise,
            ToTensor(),
            # RotationMultiple90(),
            Normalize(mean=normalization_mean, std=normalization_std)
            ]
        tst = [
            ToTensor(),
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

class RotationMultiple90:
    """Rotation counterclockwise by multiplier of 90 degree."""

    def __init__(self, multiplier: int = 1):
        """Initializes rotation by a specified multiplier of 90 degrees.

        Args:
            multiplier: multiplier of 90 degree rotation, default 1."""
        self.multiplier = multiplier

    def __call__(self, array: np.ndarray, multiplier: Optional[int] = None) -> np.ndarray:
        """Rotates array counterclockwise by 90 degree with specified multiplier.

        Args:
            array: array to be rotated.
            multiplier (optional): overwrites the multiplier set on initialization.

        Returns:
            Copy of counterclockwise rotated array."""
        if multiplier:
            self.multiplier = multiplier
        return np.rot90(array, self.multiplier, (0, 1)).copy()


class GaussianNoise():
    """Gaussian noise operation."""

    def __init__(self, std: float = 0.001, scale_value: int = 2 ** 16 - 1):
        """Initializes Gaussian noise operation.

        Args:
            std: standard deviation of Gaussian noise, default 0.001.
            scale_value: scaling value, default 65535, MAX uint16."""

        self.std = std
        self.scale_value = scale_value

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Applies Gaussian noise to the array.

        Args:
            array: array to apply Gaussian noise.

        Returns:
            Gaussian noised array."""

        normalized_array = array / self.scale_value
        noised_array = normalized_array + np.random.normal(scale=self.std, size=array.shape)
        return noised_array * self.scale_value