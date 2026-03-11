
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
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# BUFFER CLASS DEFINITIONS (Edit here to rename/add)
# ==========================================================
CLASS_CONFIG = {
    0: "Image",   # Reserved for Stencil
    1: "Bone",    # Buffer 1
    2: "Tissue",  # Buffer 2
    # 3: "Tumor",
    # 4: "Vessels",
    # ... up to 15
}

class DICOMVisualizer:
    def __init__(self, frame, window_start=0):
        self.__frame = frame
        # LAYER 0: Stencil Buffer (Image)
        self.__img_buffer_gpu = cp.array(frame.pixel_data.astype(np.int32))
        self.__h, self.__w = self.__img_buffer_gpu.shape
        self.__window_val = window_start
        
        # LAYERS 1-15: Annotation Buffers
        self.__num_buffers = 16 
        self.__mask_buffers = cp.zeros((self.__h, self.__w, self.__num_buffers), dtype=cp.uint8)
        
        # State
        self.__active_idx = 1 # Default to Bone
        self.__active_button = None
        self.__last_coord = None
        self.__temp_pts = []
        self.__roi = [0, self.__h, 0, self.__w]
        
        # Brush Config
        self.__draw_r = 10
        self.__erase_r = 40
        self.__bg_cache = None
        
        self.__setup_ui()

    def __setup_ui(self):
        self.__fig = plt.figure(figsize=(15, 10))
        self.__ax = self.__fig.add_axes([0.18, 0.12, 0.78, 0.82])
        
        # Setup Widgets (Individually managed for zero lag)
        self.__setup_widgets()
        
        # Vector Markers
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True, zorder=10)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True, zorder=11)
        self.__ax.add_patch(self.__erase_marker)
        
        # Warm up Cache
        self.__render_stencil() 
        self.__full_redraw()

        # Connect Logic
        self.__fig.canvas.mpl_connect('button_press_event', self.__on_press)
        self.__fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)
        self.__fig.canvas.mpl_connect('button_release_event', self.__on_release)
        self.__fig.canvas.mpl_connect('key_press_event', self.__on_key)

    def __setup_widgets(self):
        # Window Slider
        ax_win = self.__fig.add_axes([0.3, 0.05, 0.4, 0.02])
        self.__swin = Slider(ax_win, 'Window', 0, int(cp.asnumpy(self.__img_buffer_gpu.max()))-255, valinit=self.__window_val)
        self.__swin.on_changed(self.__update_window)

        # Radius Slider
        ax_rad = self.__fig.add_axes([0.3, 0.02, 0.4, 0.02])
        self.__srad = Slider(ax_rad, 'Eraser R', 10, 250, valinit=self.__erase_r)
        self.__srad.on_changed(self.__update_radius)

        # Class Buttons (Layers)
        self.__btns = {}
        for idx, name in CLASS_CONFIG.items():
            if idx == 0: continue # Skip image stencil
            ax_b = self.__fig.add_axes([0.02, 0.9 - (idx*0.05), 0.12, 0.04])
            btn = Button(ax_b, name)
            btn.on_clicked(lambda e, i=idx: self.__switch_buffer(i))
            self.__btns[idx] = btn
        
        self.__update_button_colors()

    def __render_stencil(self):
        """Processes Buffer 0 (Image) into a viewable stencil."""
        y1, y2, x1, x2 = self.__roi
        data = self.__img_buffer_gpu[y1:y2, x1:x2]
        buf = cp.zeros(data.shape, dtype=cp.uint8)
        mask = (data >= self.__window_val) & (data <= self.__window_val + 255)
        buf[mask] = (data[mask] - self.__window_val).astype(cp.uint8)
        self.__stencil_cpu = cp.asnumpy(buf)

    def __full_redraw(self):
        """Refreshes the high-res view and the Blit Cache."""
        y1, y2, x1, x2 = self.__roi
        self.__ax.clear()
        self.__ax.set_axis_off()
        extent = [x1, x2, y2, y1]

        # Layer 0 (Base)
        self.__ax.imshow(self.__stencil_cpu, cmap='gray', extent=extent, zorder=1)
        
        # Active Buffer Overlay
        active_data = cp.asnumpy(self.__mask_buffers[y1:y2, x1:x2, self.__active_idx])
        cm = LinearSegmentedColormap.from_list('c', [(0,0,0,0), '#00FF00'], N=2)
        self.__ax.imshow(active_data, cmap=cm, extent=extent, zorder=2, alpha=0.4)
        
        # Reset Markers
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True, zorder=10)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True, zorder=11)
        self.__ax.add_patch(self.__erase_marker)
        
        self.__ax.set_xlim(x1, x2)
        self.__ax.set_ylim(y2, y1)
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
            
            # Sub-pixel Interpolation for smooth lines
            dy, dx = curr[0] - self.__last_coord[0], curr[1] - self.__last_coord[1]
            dist = np.sqrt(dy**2 + dx**2)
            if dist > 1:
                for i in range(1, int(dist)+1):
                    self.__temp_pts.append((self.__last_coord[0] + dy*(i/dist), 
                                           self.__last_coord[1] + dx*(i/dist)))
            self.__last_coord = curr
            
            # Instant Blit Interaction
            self.__fig.canvas.restore_region(self.__bg_cache)
            if self.__active_button == 1:
                py, px = zip(*self.__temp_pts)
                self.__line_marker.set_data(px, py)
                self.__ax.draw_artist(self.__line_marker)
            else:
                self.__erase_marker.center = (event.xdata, event.ydata)
                self.__ax.draw_artist(self.__erase_marker)
            self.__fig.canvas.blit(self.__ax.bbox)

    def __on_release(self, event):
        if self.__active_button:
            # Bake to Buffer in one GPU pass
            is_er = (self.__active_button == 3)
            r = self.__erase_r if is_er else self.__draw_r
            y_g, x_g = cp.ogrid[-r:r+1, -r:r+1]
            brush = (x_g**2 + y_g**2 <= r**2).astype(cp.uint8)
            
            for y, x in self.__temp_pts:
                self.__gpu_op(int(round(y)), int(round(x)), brush, r, is_er)
            
            self.__active_button = None
            self.__full_redraw()

    def __gpu_op(self, y, x, brush, r, erase):
        y_s, y_e = max(0, y-r), min(self.__h, y+r+1)
        x_s, x_e = max(0, x-r), min(self.__w, x+r+1)
        b_slice = brush[y_s-(y-r):y_s-(y-r)+(y_e-y_s), x_s-(x-r):x_s-(x-r)+(x_e-x_s)]
        if erase: self.__mask_buffers[y_s:y_e, x_s:x_e, self.__active_idx] &= ~b_slice
        else: self.__mask_buffers[y_s:y_e, x_s:x_e, self.__active_idx] |= b_slice

    def __switch_buffer(self, idx):
        self.__active_idx = idx
        self.__update_button_colors()
        self.__full_redraw()

    def __update_button_colors(self):
        for idx, btn in self.__btns.items():
            btn.label.set_color('red' if idx == self.__active_idx else 'black')
            btn.color = 'lightgray' if idx == self.__active_idx else '0.85'

    def __update_window(self, val):
        self.__window_val = int(val)
        self.__render_stencil()
        self.__full_redraw()

    def __update_radius(self, val):
        self.__erase_r = int(val)
        self.__erase_marker.set_radius(self.__erase_r)

    def __on_key(self, event):
        if event.key == 'q':
            # Export Layers 1-15 as Binary Masks
            masks = cp.asnumpy(self.__mask_buffers[:,:,1:] > 0).astype(np.uint8)
            self.__frame.pixel_data = masks
            plt.close(self.__fig)

    def show(self):
        plt.show()
