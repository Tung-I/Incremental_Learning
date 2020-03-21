import re
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.datasets import BaseDataset
from src.data.transforms import Compose, ToTensor


class AcdcSegDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 for the segmentation task.

    Ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

    Args:
        data_split_file_path (str): 
        transforms (BoxList): The preprocessing techniques applied to the data.
        augments (BoxList): The augmentation techniques applied to the training data (default: None).
    """

    def __init__(self, data_split_file_path, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        data_split_file = pd.read_csv(data_split_file_path)
        patient_dirs = tuple(map(Path, data_split_file[data_split_file.type == self.type].path))
        self.data_paths = tuple(
            data_path
            for patient_dir in patient_dirs
            for data_path in zip(
                sorted(patient_dir.glob('**/*frame??.nii.gz')),
                sorted(patient_dir.glob('**/*frame??_gt.nii.gz'))
                # sorted(patient_dir.glob('**/*frame??.nii.gz')) # For the testing dataset.
            )
        )
        self.transforms = Compose.compose(transforms)
        self.augments = Compose.compose(augments)
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        mr_path, gt_path = self.data_paths[index]
        nib_img = nib.load(mr_path.as_posix())
        mr = nib_img.get_fdata().astype(np.float32)[..., np.newaxis]
        gt = nib.load(gt_path.as_posix()).get_fdata().astype(np.int64)[..., np.newaxis]
        input_spacing = nib_img.header['pixdim'][1:4]

        if self.type == 'train':
            transforms_kwargs = {
                'Resample': {
                    'input_spacings': (input_spacing, input_spacing),
                    'orders': (1, 0)
                },
                'Clip': {
                    'transformed': (True, False)
                },
                'MinMaxScale': {
                    'transformed': (True, False),
                }
            }
            mr, gt = self.transforms(mr, gt, **transforms_kwargs)
            mr, gt = self.augments(mr, gt)
            mr, gt = self.to_tensor(mr, gt)
        else:
            transforms_kwargs = {
                'Resample': {
                    'input_spacings': (input_spacing,),
                    'orders': (1,)
                }
            }
            mr, = self.transforms(mr, **transforms_kwargs)
            mr, gt = self.to_tensor(mr, gt)
        metadata = {'input': mr, 'target': gt}

        if self.type == 'test':
            metadata.update(affine=nib_img.affine,
                            name=re.sub(r'frame\d+', ('ED' if index % 2 == 0 else 'ES'), mr_path.name))
        return metadata

    def __len__(self):
        return len(self.data_paths)
