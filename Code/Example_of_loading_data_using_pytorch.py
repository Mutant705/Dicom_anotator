import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DICOMSegmentationDataset(Dataset):
    def __init__(self, data_dir, target_classes, vmin=0, vmax=65535):
        """
        Args:
            data_dir (str): Path to the folder containing your .npz files.
            target_classes (list of int): List of mask IDs to load (e.g., [1] for Bones).
            vmin/vmax (int/float): Windowing bounds for the raw uint16 data.
        """
        self.file_paths = glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
        self.target_classes = target_classes 
        self.vmin = vmin
        self.vmax = vmax

    def __len__(self):
        return len(self.file_paths)

    def apply_windowing(self, raw_pixels):
        # Format the uint16 data for the neural network (0.0 to 1.0)
        data = np.clip(raw_pixels.astype(np.float32), self.vmin, self.vmax)
        return (data - self.vmin) / (self.vmax - self.vmin + 1e-7)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        
        # 1. Process the Image Tensor [1, H, W]
        image_norm = self.apply_windowing(data['raw_image'])
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).float()

        # 2. Process ONLY the requested target masks [C, H, W]
        h, w = image_norm.shape
        mask_stack = np.zeros((len(self.target_classes), h, w), dtype=np.float32)
        
        for i, class_idx in enumerate(self.target_classes):
            mask_key = f"mask_{class_idx}"
            if mask_key in data:
                mask_stack[i] = data[mask_key]

        mask_tensor = torch.from_numpy(mask_stack)
        return image_tensor, mask_tensor
