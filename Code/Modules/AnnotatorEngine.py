import cupy as cp
import numpy as np
import cv2

class AnnotatorEngine:
    def __init__(self, frame_data):
        self.h, self.w = frame_data.shape
        self.raw_gpu = cp.array(frame_data.astype(cp.int32))
        self.mask_buffers = cp.zeros((self.h, self.w, 16), dtype=cp.uint8)
        self.view_stencil_cpu = None 

    def update_stencil(self, gpu_8bit_array):
        self.view_stencil_cpu = cp.asnumpy(gpu_8bit_array)

    def apply_stroke(self, y1, x1, y2, x2, r, idx, erase=False):
        dist = int(np.hypot(y2 - y1, x2 - x1))
        steps = max(dist, 1)
        y_pts = np.linspace(y1, y2, steps)
        x_pts = np.linspace(x1, x2, steps)
        for i in range(steps):
            self.apply_brush(int(y_pts[i]), int(x_pts[i]), r, idx, erase)

    def apply_brush(self, y, x, r, idx, erase=False):
        # Boundary-safe slicing logic
        y_s, y_e = max(0, y-r), min(self.h, y+r+1)
        x_s, x_e = max(0, x-r), min(self.w, x+r+1)
        br_y_s, br_y_e = y_s - (y-r), y_e - (y-r)
        br_x_s, br_x_e = x_s - (x-r), x_e - (x-r)

        y_grid, x_grid = cp.ogrid[-r:r+1, -r:r+1]
        brush_full = (x_grid**2 + y_grid**2 <= r**2).astype(cp.uint8)
        b_slice = brush_full[br_y_s:br_y_e, br_x_s:br_x_e]
        
        if erase:
            self.mask_buffers[y_s:y_e, x_s:x_e, idx] &= ~b_slice
        else:
            self.mask_buffers[y_s:y_e, x_s:x_e, idx] |= b_slice

    def fill_closed_curve(self, coords, idx):
        if len(coords) < 3: return
        temp_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array(coords, dtype=np.int32)
        cv2.fillPoly(temp_mask, [pts], 1)
        self.mask_buffers[:, :, idx] |= cp.array(temp_mask)

    def get_mask_cpu(self, idx):
        mask = cp.asnumpy(self.mask_buffers[:, :, idx])
        return (mask > 0).astype(np.uint8)

