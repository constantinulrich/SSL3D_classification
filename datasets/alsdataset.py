'''
Author: Justin Rozeboom
Data preparation of NIfTI files
'''

import json
import os
import re
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO
import shutil



class ALSData(Dataset):
    def __init__(self, root, split, fold, transform=None, use_preprocessed=True):
        super().__init__()
        """
        ALS Dataset for NIfTI files
        
        Args:
            root: Path to dataset root
            split: 'train' or 'val'
            fold: Cross-validation fold number
            transform: Data transforms to apply
            use_preprocessed: If True, look for preprocessed .b2nd files first
        """
        self.root = Path(root)
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        
        # Check for preprocessed data directory (following project pattern)
        if use_preprocessed:
            self.img_dir = self.root / "nnUNetResEncUNetLPlans_3d_fullres"
            if self.img_dir.exists():
                self._load_preprocessed_data(split, fold)
                return
        
        # Load splits and labels from JSON files
        split_file = self.root / "splits_final.json"
        label_file = self.root / "labelsTr.json"
        
        if split_file.exists() and label_file.exists():
            # Use existing split files (preferred approach)
            with open(split_file) as f:
                splits = json.load(f)
                self.img_files = splits[fold]["train" if split == "train" else "val"]
            
            with open(label_file) as f:
                labels = json.load(f)
            self.labels = [labels[i] for i in self.img_files]
        else:
            # Fallback: create splits from NIfTI files directly
            self._create_splits_from_raw_data(split)
    
    def _load_preprocessed_data(self, split, fold):
        """Load preprocessed data in .b2nd format"""
        split_file = self.root / "splits_final.json"
        label_file = self.root / "labelsTr.json"
        
        with open(split_file) as f:
            splits = json.load(f)
            self.img_files = splits[fold]["train" if split == "train" else "val"]
        
        with open(label_file) as f:
            labels = json.load(f)
        # Handle both simple label dict and nested label format
        if isinstance(list(labels.values())[0], list):
            self.labels = [labels[i][1] for i in self.img_files]
        else:
            self.labels = [labels[i] for i in self.img_files]
    
    def _create_splits_from_raw_data(self, split):
        """Create splits from raw NIfTI files (fallback method)"""
        # Look for imagesTr directory first (nnU-Net format)
        images_dir = self.root / "imagesTr"
        if images_dir.exists():
            self.data_dir = images_dir
            self.nifti_files = [f for f in os.listdir(self.data_dir) 
                              if f.endswith((".nii", ".nii.gz"))]
        else:
            # Fallback: look for NIfTI files directly in the root directory
            self.data_dir = self.root
            self.nifti_files = [f for f in os.listdir(self.data_dir) 
                      if f.endswith((".nii", ".nii.gz"))]
        
        assert len(self.nifti_files) > 0, f"No NIfTI files found in {self.data_dir}"
        
        # Process files and extract labels
        file_data = []
        for f in self.nifti_files:
            base_id = f.replace(".nii.gz", "").replace(".nii", "")
            
            if "CALSNIC1" in f or "CALSNIC2" in f or "CAPTURE" in f:
                # Class label in filename (e.g. P001, C001)
                match_p = re.search(r'P\d{3}', f)
                match_c = re.search(r'C\d{3}', f)
                if match_p:
                    label = 1  # patient
                    file_data.append({'id': base_id, 'label': label})
                elif match_c:
                    label = 0  # control
                    file_data.append({'id': base_id, 'label': label})
                else:
                    raise ValueError(f"Filename {f} does not contain expected class label (P or C)")
            else:
                raise ValueError(f"Filename {f} does not contain CALSNIC or CAPTURE")
        
        # Generate JSON files if they don't exist
        self._generate_json_files(file_data)
        
        # Now load the data using the generated JSON files
        split_file = self.root / "splits_final.json"
        label_file = self.root / "labelsTr.json"
        
        with open(split_file) as f:
            splits = json.load(f)
            # Use fold 0 as default when creating from raw data
            fold = 0
            self.img_files = splits[fold]["train" if split == "train" else "val"]
        
        with open(label_file) as f:
            labels = json.load(f)
        self.labels = [labels[i] for i in self.img_files]
    
    def _generate_json_files(self, file_data):
        """Generate labelsTr.json and splits_final.json from raw data"""
        # Create labels dictionary
        labels = {item['id']: item['label'] for item in file_data}
        
        # Save labelsTr.json
        label_file = self.root / "labelsTr.json"
        with open(label_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        # Generate cross-validation splits (3 folds)
        np.random.seed(42)  # For reproducible splits
        all_ids = [item['id'] for item in file_data]
        indices = np.random.permutation(len(all_ids))
        
        # Create 3-fold cross-validation splits
        splits = {}
        n_folds = 3
        fold_size = len(all_ids) // n_folds
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            if fold == n_folds - 1:  # Last fold gets remaining items
                end_idx = len(all_ids)
            else:
                end_idx = (fold + 1) * fold_size
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            splits[fold] = {
                "train": [all_ids[i] for i in train_indices],
                "val": [all_ids[i] for i in val_indices]
            }
        
        # Save splits_final.json
        split_file = self.root / "splits_final.json"
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Generated {label_file} with {len(labels)} labels")
        print(f"Generated {split_file} with {n_folds} folds")
        for fold in range(n_folds):
            print(f"  Fold {fold}: {len(splits[fold]['train'])} train, {len(splits[fold]['val'])} val")

    def __getitem__(self, idx):
        # Check if we're using preprocessed data
        if hasattr(self, 'img_dir') and self.img_dir.exists():
            # Load preprocessed .b2nd file
            img, _ = Blosc2IO.load(self.img_dir / (self.img_files[idx] + ".b2nd"), mode="r")
            
            if self.transform:
                img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
            else:
                img = torch.from_numpy(img[...])
        else:
            # Load NIfTI file
            if hasattr(self, 'data_dir'):
                img_path = os.path.join(self.data_dir, self.img_files[idx])
            else:
                img_path = self.root / self.img_files[idx]
            
            img = nibabel.load(img_path).get_fdata(dtype='float32')
            
            # # Resize to standard shape (96, 96, 96) with channel dimension
            # img = self._resize_image(img, (96, 96, 96), add_channel_dim=True)
            
            if self.transform:
                img = self.transform(**{"image": torch.from_numpy(img)})["image"]
            else:
                img = torch.from_numpy(img)

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)
    
    def _resize_image(self, img, new_shape, add_channel_dim=True):
        """
        Resize image to new_shape with trilinear interpolation.
        eg. SwinUNETR requires input shape divisible by 32
        """
        zoom_factors = [n / o for n, o in zip(new_shape, img.shape)]
        resized_img = zoom(img, zoom_factors, order=1)
        if add_channel_dim:
            resized_img = resized_img[None, ...]
        return resized_img




class ALSDataModule(BaseDataModule):
    def __init__(self, **params):
        super(ALSDataModule, self).__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = ALSData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = ALSData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )


def make_file_structure(input_files_dir, dataset_root):
    """
    Create the directory structure for the ALS dataset.
    The structure is based on the nnU-Net format.
    Args:
        input_files_dir: Path to the directory containing the .nii files.
        dataset_root: Path to the root directory where the dataset structure will be created.
    """
    # TODO: handle V2 / run-02 files
    # TODO: include multimodal support by naming as NAME_0000 (T1), NAME_0001 (T2), etc. labels json holds just NAME

    dataset_root = Path(dataset_root)
    images_tr = dataset_root / "imagesTr"
    images_ts = dataset_root / "imagesTs"   # will be empty for now
    try:
        images_tr.mkdir(parents=True)
        images_ts.mkdir(parents=True)
    except FileExistsError as e:
        print(f"{images_tr} or {images_ts} already exist. Delete them and rerun this function.")
        raise e

    # Copy NIfTI files to the imagesTr directory
    print(f"Copying NIfTI files from {input_files_dir} to {images_tr}")
    for file in os.listdir(input_files_dir):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            src = Path(input_files_dir) / file
            dst = images_tr / file
            shutil.copy2(src, dst)

    print(f"nnU-Net-style Dataset structure created at {dataset_root}")

    # For classification, we need to make LabelsTr.json
    # Prepare any missing label info
    for f in os.listdir(images_tr):
        match_pc = re.search(r'_(P|C)(\d{3})', f)
        if not match_pc:
            response = input(f"File '{f}' does not match expected CALSNIC naming (missing _P### or _C###). Run the file labelling function to fix? (Y/N): ")
            if response.strip().lower() == 'y':
                csv_file = "/home/alslab/data/CAPTURE_DQT_26062025.csv"
                label_capture_from_csv(csv_file, images_tr)
            else:
                raise KeyboardInterrupt("Cancelled")
        
    labels = {}
    for f in os.listdir(images_tr):
        base_id = f.replace(".nii.gz", "").replace(".nii", "")
        if re.search(r'_P\d{3}', f):
            labels[base_id] = 1
        elif re.search(r'_C\d{3}', f):
            labels[base_id] = 0
        else:
            raise ValueError(f"File '{f}' does not match expected CALSNIC naming (_P### or _C###)")

    label_file = dataset_root / "labelsTr.json"
    with open(label_file, 'w') as lf:
        json.dump(labels, lf, indent=2)
    print(f"Generated {label_file} with {len(labels)} labels")

    


def label_capture_from_csv(csv_file, data_dir):
    '''
    Alter the filenames in data_dir to follow CALSNIC naming conventions by adding patient/control labels
    TODO Check metadata of CALSNIC to ensure consistent labelling
    '''

    df = pd.read_csv(csv_file)
    files = [f for f in os.listdir(data_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    assert len(files) > 0, f"No NIfTI files found in {data_dir}"

    count = 0
    for f in files:
        pscid = re.search(r'CAPTURE.*_(\d{3})', f)
        if pscid:
            pscid = int(pscid.group(1)[1:])
            row_df = df.loc[df['PSCID'] == f"CAPT{pscid:07}"]
            if row_df.empty:
                raise ValueError(f"PSCID {pscid} not found in CSV file")
            else:
                row = df.iloc[0]
                if row['05_Status'] == 'Patient' and (row['06_Diagnosis'] == 'ALS' or 
                                                      row['06_Diagnosis'] == 'ALSFTD'):
                    label = 'P'
                elif row['05_Status'] == 'Control':
                    label = 'C'
                else:
                    raise ValueError(f"Cannot classify {row['05_Status']} with Diagnosis: {row['06_Diagnosis']} for pscid {pscid}")
                
                # Rename files
                new_name = re.sub(r'_(\d{3})', f'_{label}{pscid:03}', f)
                os.rename(os.path.join(data_dir, f), os.path.join(data_dir, new_name))
                count += 1

    total_capture = len([f for f in os.listdir(data_dir) if "CAPTURE" in f])
    print(f"Renamed {count} files (out of {total_capture} CAPTURE files) in {data_dir} with class labels.")

