
"""
MODULE: Display_and_Anotator_gpu_accelrated.py
SYSTEM: NVIDIA RTX 4070 Optimized (requires cupy-cuda12x)
TASKS DONE:
    - GPU acceleration for 3043x3043 pixel operations.
    - Continuous Bresenham-style line interpolation.
    - Private encapsulation of all logic.
    - 'q' key transfers VRAM mask back to CPU RAM and exits.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# BUFFER CLASS DEFINITIONS (Edit names here)
# ==========================================================
CLASS_CONFIG = {
    1: "Bone",
    2: "Tissue",
    3: "Tumor",
    4: "Vessels",
    # You can add up to 15 classes here
}

class DICOMVisualizer:
    def __init__(self, frame_obj: 'DICOMFrame'):
        """
        Annotation UI using GPU buffers for high-performance drawing.
        :param frame_obj: The DICOMFrame object from Data_extractor
        """
        self.frame = frame_obj
        
        # Load raw data to GPU (Buffer 0 Placeholder)
        self.__raw_gpu = cp.array(self.frame.pixel_data.astype(np.int32))
        self.__h, self.__w = self.__raw_gpu.shape
        
        # Buffer initialization: 16 total slots (1 image stencil + 15 classes)
        # Using uint8 to save VRAM on 3043x3043 arrays
        self.__mask_buffers = cp.zeros((self.__h, self.__w, 16), dtype=cp.uint8)
        
        # UI State
        self.__active_idx = 1
        self.__window_val = int(cp.asnumpy(self.__raw_gpu.min()))
        self.__draw_r = 10
        self.__erase_r = 40
        self.__active_button = None
        self.__temp_pts = []
        
        self.__setup_ui()

    def __setup_ui(self):
        self.__fig = plt.figure(figsize=(14, 10))
        self.__ax = self.__fig.add_axes([0.18, 0.12, 0.75, 0.82])
        
        # Window Slider (Updates Buffer 0 Display)
        ax_win = self.__fig.add_axes([0.3, 0.05, 0.4, 0.02])
        self.__swin = Slider(ax_win, 'Window', 
                             int(cp.asnumpy(self.__raw_gpu.min())), 
                             int(cp.asnumpy(self.__raw_gpu.max())), 
                             valinit=self.__window_val)
        self.__swin.on_changed(self.__update_window)

        # Eraser Radius Slider
        ax_rad = self.__fig.add_axes([0.3, 0.02, 0.4, 0.02])
        self.__srad = Slider(ax_rad, 'Eraser R', 10, 300, valinit=self.__erase_r)
        self.__srad.on_changed(self.__update_radius)

        # Class Selection Buttons
        self.__btns = {}
        for idx, name in CLASS_CONFIG.items():
            ax_b = self.__fig.add_axes([0.02, 0.9 - (idx*0.05), 0.12, 0.04])
            btn = Button(ax_b, name)
            btn.on_clicked(lambda e, i=idx: self.__switch_buffer(i))
            self.__btns[idx] = btn

        # Interactive Drawing Markers
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True, zorder=10)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True, zorder=11)
        self.__ax.add_patch(self.__erase_marker)

        self.__update_button_ui()
        self.__render_stencil()
        self.__refresh_view()

        # Connect Matplotlib Events
        self.__fig.canvas.mpl_connect('button_press_event', self.__on_press)
        self.__fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)
        self.__fig.canvas.mpl_connect('button_release_event', self.__on_release)
        self.__fig.canvas.mpl_connect('key_press_event', self.__on_key)

    def __render_stencil(self):
        """Processes Buffer 0 for display view based on window value."""
        data = self.__raw_gpu
        buf = cp.zeros(data.shape, dtype=cp.uint8)
        mask = (data >= self.__window_val) & (data <= self.__window_val + 255)
        buf[mask] = (data[mask] - self.__window_val).astype(cp.uint8)
        self.__stencil_cpu = cp.asnumpy(buf)

    def __refresh_view(self):
        """Full re-render of background for blitting."""
        self.__ax.clear()
        self.__ax.set_axis_off()
        # Layer 0: Stencil
        self.__ax.imshow(self.__stencil_cpu, cmap='gray', zorder=1)
        
        # Active Layer: Mask
        mask = cp.asnumpy(self.__mask_buffers[:, :, self.__active_idx])
        cm = LinearSegmentedColormap.from_list('c', [(0,0,0,0), '#00FF00'], N=2)
        self.__ax.imshow(mask, cmap=cm, zorder=2, alpha=0.4)
        
        # Markers
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True)
        self.__ax.add_patch(self.__erase_marker)
        
        self.__fig.canvas.draw()
        self.__bg_cache = self.__fig.canvas.copy_from_bbox(self.__ax.bbox)

    def __on_press(self, event):
        if event.inaxes == self.__ax:
            self.__active_button = event.button
            self.__last_coord = (event.ydata, event.xdata)
            self.__temp_pts = [self.__last_coord]

    def __on_motion(self, event):
        if self.__active_button and event.inaxes == self.__ax:
            curr = (event.ydata, event.xdata)
            # Sub-pixel interpolation
            dist = np.hypot(curr[0]-self.__last_coord[0], curr[1]-self.__last_coord[1])
            steps = max(int(dist), 1)
            for i in range(1, steps + 1):
                self.__temp_pts.append((self.__last_coord[0] + (curr[0]-self.__last_coord[0])*(i/steps),
                                       self.__last_coord[1] + (curr[1]-self.__last_coord[1])*(i/steps)))
            self.__last_coord = curr
            
            self.__fig.canvas.restore_region(self.__bg_cache)
            if self.__active_button == 1: # Draw
                py, px = zip(*self.__temp_pts)
                self.__line_marker.set_data(px, py)
                self.__ax.draw_artist(self.__line_marker)
            elif self.__active_button == 3: # Erase
                self.__erase_marker.center = (event.xdata, event.ydata)
                self.__ax.draw_artist(self.__erase_marker)
            self.__fig.canvas.blit(self.__ax.bbox)

    def __on_release(self, event):
        if self.__active_button:
            is_er = (self.__active_button == 3)
            r = self.__erase_r if is_er else self.__draw_r
            y_g, x_g = cp.ogrid[-r:r+1, -r:r+1]
            brush = (x_g**2 + y_g**2 <= r**2).astype(cp.uint8)
            
            for y, x in self.__temp_pts:
                self.__gpu_op(int(round(y)), int(round(x)), brush, r, is_er)
            
            self.__active_button = None
            self.__refresh_view()

    def __gpu_op(self, y, x, brush, r, erase):
        y_s, y_e = max(0, y-r), min(self.__h, y+r+1)
        x_s, x_e = max(0, x-r), min(self.__w, x+r+1)
        b_slice = brush[y_s-(y-r):y_s-(y-r)+(y_e-y_s), x_s-(x-r):x_s-(x-r)+(x_e-x_s)]
        if erase: self.__mask_buffers[y_s:y_e, x_s:x_e, self.__active_idx] &= ~b_slice
        else: self.__mask_buffers[y_s:y_e, x_s:x_e, self.__active_idx] |= b_slice

    def __switch_buffer(self, idx):
        self.__active_idx = idx
        self.__update_button_ui()
        self.__refresh_view()

    def __update_button_ui(self):
        for idx, btn in self.__btns.items():
            btn.label.set_color('red' if idx == self.__active_idx else 'black')

    def __update_window(self, val):
        self.__window_val = int(val)
        self.__render_stencil()
        self.__refresh_view()

    def __update_radius(self, val):
        self.__erase_r = int(val)
        self.__erase_marker.set_radius(self.__erase_r)

    def __on_key(self, event):
        if event.key == 'enter':
            # Injection: Export GPU masks to the frame object
            # Stores as a list of 15 numpy arrays
            self.frame.masks = [cp.asnumpy(self.__mask_buffers[:,:,i] > 0) for i in range(1, 16)]
            self.frame.class_mapping = CLASS_CONFIG
            self.frame.is_annotated = True
            plt.close(self.__fig)
        elif event.key == 'q':
            plt.close(self.__fig)

    def show(self):
        plt.show()
