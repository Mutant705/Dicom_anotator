"""
MODULE: Display_and_Anotate.py (GPU Accelerated with CuPy)
SYSTEM: NVIDIA RTX 4070 Optimized
"""

import numpy as np
import cupy as cp  # GPU acceleration
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap

class DICOMVisualizer:
    def __init__(self, frame, window_start=0):
        self.__frame = frame
        self.__raw_data_cpu = frame.pixel_data.astype(np.int32)
        
        # 1. Move raw data to GPU for faster windowing calculations
        self.__raw_data_gpu = cp.array(self.__raw_data_cpu)
        self.__window_val = window_start
        
        self.__class_names = ["Tumor", "Tissue", "Bone"]
        
        # 2. Store the 3D Mask on GPU VRAM
        self.__mask_3d_gpu = cp.zeros((*self.__raw_data_cpu.shape, len(self.__class_names)), dtype=cp.uint8)
        
        self.__active_idx = 0
        self.__is_eraser = False
        self.__visible = [True] * len(self.__class_names)
        self.__is_drawing = False
        self.__last_x, self.__last_y = None, None
        
        # Pre-calculate the circular brush on GPU
        self.__r = 3
        y_g, x_g = cp.ogrid[-self.__r : self.__r+1, -self.__r : self.__r+1]
        self.__brush_gpu = (x_g**2 + y_g**2 <= self.__r**2).astype(cp.uint8)
        
        self.__setup_ui()

    def __calculate_display(self, win_start):
        """GPU Accelerated Windowing."""
        if str(win_start).lower() == "normalize":
            d_min, d_max = self.__raw_data_gpu.min(), self.__raw_data_gpu.max()
            if d_max == d_min: return np.zeros_like(self.__raw_data_cpu, dtype=np.uint8)
            buf_gpu = ((self.__raw_data_gpu - d_min) / (d_max - d_min) * 255).astype(cp.uint8)
        else:
            ws = int(win_start)
            buf_gpu = cp.zeros_like(self.__raw_data_gpu, dtype=cp.uint8)
            mask = (self.__raw_data_gpu >= ws) & (self.__raw_data_gpu <= ws + 255)
            buf_gpu[mask] = (self.__raw_data_gpu[mask] - ws).astype(cp.uint8)
        
        # Transfer only the 8-bit display buffer back to CPU for Matplotlib
        return cp.asnumpy(buf_gpu)

    def __apply_brush(self, x_f, y_f):
        """GPU-based bitwise mask operations."""
        if not self.__visible[self.__active_idx]:
            return
        
        x, y = int(round(x_f)), int(round(y_f))
        val = 0 if self.__is_eraser else 1
        
        img_h, img_w = self.__raw_data_cpu.shape
        y_s, y_e = max(0, y - self.__r), min(img_h, y + self.__r + 1)
        x_s, x_e = max(0, x - self.__r), min(img_w, x + self.__r + 1)
        
        b_y_s = y_s - (y - self.__r)
        b_y_e = b_y_s + (y_e - y_s)
        b_x_s = x_s - (x - self.__r)
        b_x_e = b_x_s + (x_e - x_s)
        
        b_slice = self.__brush_gpu[b_y_s:b_y_e, b_x_s:b_x_e]
        
        # Perform Bitwise OR/AND on GPU (RTX 4070 will handle this instantly)
        if val == 1:
            self.__mask_3d_gpu[y_s:y_e, x_s:x_e, self.__active_idx] |= b_slice
        else:
            self.__mask_3d_gpu[y_s:y_e, x_s:x_e, self.__active_idx] &= ~b_slice
        
        # Only transfer the slice of the mask currently being edited back to CPU for the overlay
        mask_cpu = cp.asnumpy(self.__mask_3d_gpu[:, :, self.__active_idx])
        self.__overlays[self.__active_idx].set_data(mask_cpu)

    def __on_key(self, event):
        """Pull data back to CPU and save on Exit."""
        if event.key == 'q':
            # Transfer the final 3D mask from VRAM to System RAM
            self.__frame.pixel_data = cp.asnumpy(self.__mask_3d_gpu)
            plt.close(self.__fig)

    # ... [Keep __setup_ui, __on_press, __on_release, __on_motion, __draw_line, show as before] ...
