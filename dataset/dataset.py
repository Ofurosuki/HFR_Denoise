import os
import torch
from torch.utils.data import Dataset
import blosc2
import numpy as np


class HistMatrixDataset(Dataset):
    def __init__(self, root_path: str, split: str, dataset_name: str, scan_type: str, sync_angle: float,
                 transform=None):
        """
        Args:
            root_path (string): Path to the absolute root of the dataset.
            split (string): 'train', 'val', or 'test'.
            dataset_name (string): e.g., 'nuscenes'.
            scan_type (string): e.g., 'horizontal'.
            sync_angle (float): The synchronization angle.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        sync_angle_str = str(sync_angle).replace('.', '_')

        self.dataset_path = os.path.join(root_path, split, dataset_name, scan_type, sync_angle_str)

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.sample_dirs = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {self.dataset_path}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir_name = self.sample_dirs[idx]
        sample_path = os.path.join(self.dataset_path, sample_dir_name)

        signal_path = os.path.join(sample_path, 'signal.bl2')
        labels_path = os.path.join(sample_path, 'labels.bl2')

        try:
            with open(signal_path, 'rb') as f:
                signal_data = blosc2.unpack_array(f.read())

            with open(labels_path, 'rb') as f:
                labels_data = blosc2.unpack_array(f.read())

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found in {sample_path}: {e}")

        # Convert numpy arrays to torch tensors
        signal_tensor = torch.from_numpy(signal_data.astype(np.float32))
        labels_tensor = torch.from_numpy(labels_data.astype(np.int64))

        sample = {'signal': signal_tensor, 'labels': labels_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample
