import re
import nibabel as nib
import numpy as np
import pandas as pd
import csv
import torch
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class IncrementalDataset(BaseDataset):
    def __init__(self, data_dir, csv_name, chosen_index, chosen_label, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.csv_name = csv_name
        self.chosen_index = chosen_index
        self.chosen_label = chosen_label
        self.transforms = Compose.compose(transforms)
        self.augments = Compose.compose(augments)
        self.to_tensor = ToTensor()
        if self.type != 'Testing':
            self.data_dir = self.data_dir / Path('train')
        else:
            self.data_dir = self.data_dir / Path('test')
        self.class_folder_path = sorted([_dir for _dir in self.data_dir.iterdir() if _dir.is_dir()])
        self.data_paths = []

        for i, idx in enumerate(self.chosen_index):
            folder_path = self.class_folder_path[idx]
            if self.type != 'Testing':
                csv_path = str(folder_path / csv_name)
                with open(csv_path, 'r', newline='') as csvfile:
                    rows = csv.reader(csvfile)
                    for _path, _type in rows:
                        if self.type==_type:
                            self.data_paths.append((_path, self.chosen_label[i]))
            else:
                file_paths = list(folder_path.glob('*.npy'))
                for _path in file_paths:
                    self.data_paths.append((_path, self.chosen_label[i]))

    def __getitem__(self, index):
        data_path, gt = self.data_paths[index]
        img = np.load(data_path).astype(np.float32)

        gt = torch.as_tensor(np.asarray([gt]).astype(np.int16))

        if self.type == 'Training':
            transforms_kwargs = {}
            img, = self.transforms(img, **transforms_kwargs)
            img, = self.augments(img)
            img, = self.to_tensor(img)
        else:
            transforms_kwargs = {}
            img, = self.transforms(img, **transforms_kwargs)
            img, = self.to_tensor(img)

        img = img.permute(2, 0, 1).float().contiguous()
        gt = gt.long().contiguous()
        return {'inputs':img, 'targets':gt}

    def __len__(self):
        return len(self.data_paths)
