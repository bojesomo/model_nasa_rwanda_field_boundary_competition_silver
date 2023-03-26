import kornia as K
from kornia.augmentation import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation)
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
import numpy as np
import pandas as pd
import pickle


class BoundaryDataset(Dataset):
    """NASA Field Boundary Dataset. Read images
    """

    def __init__(
            self,
            chip_ids,
            repeats=1,
            include_extent=False,
            include_background=False,
            include_distance=False,
            use_visible_bands_only=False,
            file_path_common="data/train",
            normalize=False,
    ):

        self.chip_ids = chip_ids
        self.repeats = repeats
        self.file_path_common = file_path_common

        self.include_extent = include_extent
        self.include_background = include_background

        self.normalize = normalize

        self.ids_mask = [0]
        if include_extent:
            self.ids_mask.append(1)
        if include_background:
            self.ids_mask.append(2)
        if include_distance:
            self.ids_mask.append(3)

        self.use_visible_bands_only = use_visible_bands_only

        if normalize:
            base = os.path.join(*file_path_common.rstrip(os.sep).split(os.sep)[:-1])
            df_statistics = pd.read_csv(f'{base}/statistics.csv', index_col=0)
            self._mean = df_statistics.values[0].reshape(-1, 1, 1).astype('float32')
            self._std = df_statistics.values[1].reshape(-1, 1, 1).astype('float32')
        else:
            self._mean = np.zeros((4 * 6, 1, 1)).astype('float32')
            self._std = np.ones((4 * 6, 1, 1)).astype('float32')

        self.load_inputs()

    def normalize_image(self, image):
        return (image - self._mean) / self._std

    def load_inputs(self):
        self.images = []
        if 'train' in self.file_path_common:
            self.masks = []
        for index in range(len(self.chip_ids)):
            image, mask = self.load_input(index)

            image = self.normalize_image(image)

            if self.use_visible_bands_only:
                image = image[~np.in1d(range(24), np.arange(24)[3::4])]

            self.images.append(image)
            if 'train' in self.file_path_common:
                self.masks.append(mask)

    def load_input(self, index):
        chip_id = self.chip_ids[index]
        
        # read data
        image_path = f"{self.file_path_common}/fields/{chip_id}.pkl"
        image = self.load_file(image_path)

        mask_path = f"{self.file_path_common}/masks/{chip_id}.pkl"
        mask = self.load_file(mask_path)
        if mask is not None:
            mask = mask[self.ids_mask]

        return image, mask

    def load_file(self, filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            data = data.transpose(2, 0, 1).astype('float32')
        return data

    def __getitem__(self, index):
        index = index % len(self.chip_ids)

        chip_id = self.chip_ids[index]
        image = self.images[index]
        sample = dict(
            chip_id=chip_id,
            image=image,
        )
        if 'train' in self.file_path_common:
            mask = self.masks[index]
            sample['mask'] = mask

        return sample

    def __len__(self):
        return len(self.chip_ids) * self.repeats


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = K.augmentation.AugmentationSequential(
            RandomHorizontalFlip(p=0.7),
            RandomVerticalFlip(p=0.7),
            RandomRotation(degrees=(90, 90), p=0.7),
            data_keys=['input', 'mask'],
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, sample):
        sample['image'], sample['mask'] = self.transforms(sample['image'], sample['mask'])  # BxCxHxW

        return sample


class BoundaryData(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 seed=42,
                 workers=4,
                 repeats=1,
                 normalize=True,
                 use_visible_bands_only=False,
                 include_extent=False,
                 include_background=False,
                 include_distance=False,
                 data_root='data'
                 ):
        super().__init__()
        self.class_weight = None
        self.workers = workers
        self.batch_size = batch_size
        self.repeats = repeats
        self.normalize = normalize
        self.use_visible_bands_only = use_visible_bands_only
        self.seed = seed
        self.include_extent = include_extent
        self.include_background = include_background
        self.include_distance = include_distance
        self.data_root = data_root

        self.train_dst = self.test_dst = self.val_dst = None
        self.train_fold = self.val_fold = None
        self.train_dims = self.val_dims = None

        self.transform = DataAugmentation()  # per batch augmentation_kornia

    def setup(self, stage=None):
        df_train = pd.read_csv(os.path.join(self.data_root, 'train.csv'), converters = {'chip_id': str})
        df_test = pd.read_csv(os.path.join(self.data_root, 'test.csv'), converters = {'chip_id': str})
        
        test_path = f"{self.data_root}/test/"
#         print(test_path)
        self.test_dst = BoundaryDataset(
            df_test.chip_id.unique(),
            repeats=1,
            normalize=self.normalize,
            include_extent=self.include_extent,
            include_background=self.include_background,
            include_distance=self.include_distance,
            use_visible_bands_only=self.use_visible_bands_only,
            file_path_common=test_path,
        )
        
        train_path = f"{self.data_root}/train/"
#         print(train_path)
        self.train_dst = BoundaryDataset(
            df_train.chip_id.unique(),
            repeats=self.repeats,
            normalize=self.normalize,
            include_extent=self.include_extent,
            include_background=self.include_background,
            include_distance=self.include_distance,
            use_visible_bands_only=self.use_visible_bands_only,
            file_path_common=train_path,
        )
        self.train_dims = len(self.train_dst)

    def train_dataloader(self):
        return DataLoader(self.train_dst, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          pin_memory=True, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dst, batch_size=self.batch_size, shuffle=False,
                          pin_memory=False, num_workers=self.workers)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            batch = self.transform(batch)  # => we perform GPU/Batched data augmentation
        return batch

