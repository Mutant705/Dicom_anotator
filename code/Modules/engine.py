import cupy as cp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class ImageEngine:
    def __init__(self, frame_obj):
        # Load raw data to GPU
        self.raw_gpu = cp.array(frame_obj.pixel_data.astype(cp.float32))
        self.h, self.w = self.raw_gpu.shape
        
        # Initialize Masks (16 slots)
        self.mask_buffers = cp.zeros((self.h, self.w, 16), dtype=cp.uint8)
        self.active_mask_idx = 1
        
        # Current Display State
        self.stencil_cpu = np.zeros((self.h, self.w), dtype=np.uint8)
        self.mask_cm = LinearSegmentedColormap.from_list('c', [(0,0,0,0), '#00FF00'], N=2)

    def normalize(self, mode, v1, v2):
        """Perform GPU normalization based on parameters."""
        data = self.raw_gpu
        
        if mode == "Linear":
            res = cp.clip(data, v1, v2)
            res = (res - v1) / (v2 - v1 + 1e-7)
        elif mode == "Skewed":
            res = 1.0 / (1.0 + cp.exp(-(data - v1) / (v2 + 1e-7)))
        elif mode == "Z-Score":
            mean, std = cp.mean(data), cp.std(data)
            res = (data - (mean + v2)) / (std + 1e-7)
            res = cp.clip(res, -v1, v1)
            res = (res + v1) / (2 * v1)
        elif mode == "Log":
            res = cp.log1p(cp.maximum(0, data * v1 + v2))
            res = (res - res.min()) / (res.max() - res.min() + 1e-7)
        else:
            res = data

        self.stencil_cpu = cp.asnumpy((res * 255).astype(cp.uint8))
        return self.stencil_cpu

    def get_active_mask(self):
        return cp.asnumpy(self.mask_buffers[:, :, self.active_mask_idx])
