import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# BUFFER CLASS DEFINITIONS
# ==========================================================
CLASS_CONFIG = {
    1: "Bones",
    2: "Left_peural_cavity",
    3: "Right_peural_cavity",
    4: "Mediastinum",
    5: "Gastric_cavity",
}

class DICOMVisualizer:
    def __init__(self, frame_obj: 'DICOMFrame'):
        self.frame = frame_obj
        
        # Load raw data to GPU
        self.__raw_gpu = cp.array(self.frame.pixel_data.astype(np.int32))
        self.__h, self.__w = self.__raw_gpu.shape
        
        # Initialize/Load Masks
        if hasattr(self.frame, 'masks') and len(self.frame.masks) > 0:
            self.__mask_buffers = cp.zeros((self.__h, self.__w, 16), dtype=cp.uint8)
            for i, m in enumerate(self.frame.masks):
                self.__mask_buffers[:, :, i+1] = cp.array(m).astype(cp.uint8)
        else:
            self.__mask_buffers = cp.zeros((self.__h, self.__w, 16), dtype=cp.uint8)
        
        # UI State
        self.__active_idx = 1
        
        # New Window Range State
        self.__win_min = int(cp.asnumpy(self.__raw_gpu.min()))
        self.__win_max = int(cp.asnumpy(self.__raw_gpu.max()))
        
        self.__draw_r = 10
        self.__erase_r = 40
        self.__active_button = None
        self.__temp_pts = []
        self.status = "stay"
        
        self.__setup_ui()

    def __setup_ui(self):
        self.__fig = plt.figure(figsize=(15, 10))
        self.__ax = self.__fig.add_axes([0.18, 0.15, 0.75, 0.8])
        
        # --- NEW TEXT BOXES FOR RANGE ---
        ax_min = self.__fig.add_axes([0.3, 0.08, 0.15, 0.03])
        self.__text_min = TextBox(ax_min, 'Range Min: ', initial=str(self.__win_min))
        self.__text_min.on_submit(self.__update_range)

        ax_max = self.__fig.add_axes([0.55, 0.08, 0.15, 0.03])
        self.__text_max = TextBox(ax_max, 'Range Max: ', initial=str(self.__win_max))
        self.__text_max.on_submit(self.__update_range)

        # --- CLASS BUTTONS (LEFT) ---
        self.__btns = {}
        for idx, name in CLASS_CONFIG.items():
            ax_b = self.__fig.add_axes([0.02, 0.9 - (idx*0.05), 0.12, 0.04])
            btn = Button(ax_b, name)
            btn.on_clicked(lambda e, i=idx: self.__switch_buffer(i))
            self.__btns[idx] = btn

        # --- NAVIGATION BUTTONS (BOTTOM) ---
        ax_prev = self.__fig.add_axes([0.18, 0.02, 0.12, 0.04])
        self.__btn_prev = Button(ax_prev, '<< Previous')
        self.__btn_prev.on_clicked(self.__on_prev)

        ax_del = self.__fig.add_axes([0.45, 0.02, 0.12, 0.04])
        ax_del.set_facecolor('mistyrose') 
        self.__btn_del = Button(ax_del, 'DELETE FRAME')
        self.__btn_del.label.set_color('red')
        self.__btn_del.on_clicked(self.__on_delete)

        ax_next = self.__fig.add_axes([0.81, 0.02, 0.12, 0.04])
        self.__btn_next = Button(ax_next, 'Next >>')
        self.__btn_next.on_clicked(self.__on_next)

        # --- MARKERS ---
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True, zorder=10)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True, zorder=11)
        self.__ax.add_patch(self.__erase_marker)

        self.__update_button_ui()
        self.__render_stencil()
        self.__refresh_view()

        # Connections
        self.__fig.canvas.mpl_connect('button_press_event', self.__on_press)
        self.__fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)
        self.__fig.canvas.mpl_connect('button_release_event', self.__on_release)
        self.__fig.canvas.mpl_connect('key_press_event', self.__on_key)

    def __update_range(self, text):
        """Validates and updates the normalization window."""
        try:
            val_min = int(self.__text_min.text)
            val_max = int(self.__text_max.text)
            if val_max > val_min:
                self.__win_min = val_min
                self.__win_max = val_max
                self.__render_stencil()
                self.__refresh_view()
        except ValueError:
            pass # Ignore non-integer input

    def __render_stencil(self):
        """Normalize GPU raw data to 0-255 based on custom Min/Max range."""
        data = self.__raw_gpu
        # Clip data to the user-defined range
        clipped = cp.clip(data, self.__win_min, self.__win_max)
        # Normalize: (x - min) / (max - min) * 255
        # Use float math for precision before converting back to uint8
        norm = (clipped - self.__win_min) / (self.__win_max - self.__win_min) * 255
        self.__stencil_cpu = cp.asnumpy(norm.astype(cp.uint8))

    def __refresh_view(self):
        self.__ax.clear()
        self.__ax.set_axis_off()
        self.__ax.imshow(self.__stencil_cpu, cmap='gray', vmin=0, vmax=255, zorder=1)
        
        mask = cp.asnumpy(self.__mask_buffers[:, :, self.__active_idx])
        cm = LinearSegmentedColormap.from_list('c', [(0,0,0,0), '#00FF00'], N=2)
        self.__ax.imshow(mask, cmap=cm, zorder=2, alpha=0.4)
        
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True)
        self.__ax.add_patch(self.__erase_marker)
        self.__fig.canvas.draw()
        self.__bg_cache = self.__fig.canvas.copy_from_bbox(self.__ax.bbox)

    # ... [Keep __on_prev, __on_next, __on_delete, __save_to_obj as they were] ...

    def __on_prev(self, event):
        self.__save_to_obj()
        self.status = "prev"
        plt.close(self.__fig)

    def __on_next(self, event):
        self.__save_to_obj()
        self.status = "next"
        plt.close(self.__fig)

    def __on_delete(self, event):
        self.status = "delete"
        plt.close(self.__fig)

    def __save_to_obj(self):
        self.frame.masks = [cp.asnumpy(self.__mask_buffers[:,:,i] > 0) for i in range(1, 16)]
        self.frame.class_mapping = CLASS_CONFIG
        self.frame.is_annotated = True

    # ... [Keep interaction logic: __on_press, __on_motion, __on_release, __gpu_op, __switch_buffer, __update_button_ui] ...

    def __on_press(self, event):
        if event.inaxes == self.__ax:
            self.__active_button = event.button
            self.__last_coord = (event.ydata, event.xdata)
            self.__temp_pts = [self.__last_coord]

    def __on_motion(self, event):
        if self.__active_button and event.inaxes == self.__ax:
            curr = (event.ydata, event.xdata)
            dist = np.hypot(curr[0]-self.__last_coord[0], curr[1]-self.__last_coord[1])
            steps = max(int(dist), 1)
            for i in range(1, steps + 1):
                self.__temp_pts.append((self.__last_coord[0] + (curr[0]-self.__last_coord[0])*(i/steps),
                                       self.__last_coord[1] + (curr[1]-self.__last_coord[1])*(i/steps)))
            self.__last_coord = curr
            self.__fig.canvas.restore_region(self.__bg_cache)
            if self.__active_button == 1:
                py, px = zip(*self.__temp_pts)
                self.__line_marker.set_data(px, py)
                self.__ax.draw_artist(self.__line_marker)
            elif self.__active_button == 3:
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

    def __on_key(self, event):
        if event.key == 'enter':
            self.__on_next(None)
        elif event.key == 'q':
            self.status = "quit"
            plt.close(self.__fig)

    def show(self):
        plt.show()
        return self.status
