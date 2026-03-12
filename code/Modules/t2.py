import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.colors import LinearSegmentedColormap

# ==========================================================
# CONFIGURATION
# ==========================================================
CLASS_CONFIG = {
    1: "Bones",
    2: "Left_pleural_cavity",
    3: "Right_peural_cavity",
    4: "Mediastinum",
    5: "Gastric_cavity",
}

class DICOMVisualizer:
    def __init__(self, frame_obj):
        self.frame = frame_obj
        
        # Load raw data to GPU
        self.__raw_gpu = cp.array(self.frame.pixel_data.astype(cp.float32))
        self.__h, self.__w = self.__raw_gpu.shape
        
        # Initialize Masks
        self.__mask_buffers = cp.zeros((self.__h, self.__w, 16), dtype=cp.uint8)
        self.__active_idx = 1
        
        # --- NORMALIZATION STATE ---
        self.norm_mode = "Linear"
        self.__params = {
            "Linear":  {"v1": 1000.0, "v2": 3500.0, "l1": "Min: ",    "l2": "Max: "},
            "Skewed":  {"v1": 1800.0, "v2": 300.0,  "l1": "Center: ", "l2": "Width: "},
            "Z-Score": {"v1": 2.0,    "v2": 0.0,    "l1": "Clip Std:", "l2": "Shift: "},
            "Log":     {"v1": 1.0,    "v2": 1.0,    "l1": "Gain: ",   "l2": "Offset: "}
        }
        
        self.__draw_r = 10
        self.__erase_r = 40
        self.__active_button = None
        self.__temp_pts = []
        self.status = "stay"
        
        self.__setup_ui()

    def __setup_ui(self):
        self.__fig = plt.figure(figsize=(16, 10))
        self.__ax = self.__fig.add_axes([0.18, 0.15, 0.70, 0.8])
        
        # --- NORMALIZATION MODE BUTTONS ---
        self.__norm_btns = {}
        modes = [("Linear", 0.8), ("Skewed", 0.75), ("Z-Score", 0.7), ("Log", 0.65)]
        for name, pos in modes:
            ax_n = self.__fig.add_axes([0.89, pos, 0.08, 0.04])
            btn = Button(ax_n, name)
            # Clicking mode ONLY updates UI/internal state now
            btn.on_clicked(lambda e, n=name: self.__set_norm_mode_ui(n))
            self.__norm_btns[name] = btn

        # --- DYNAMIC TEXT BOXES ---
        self.__ax_v1 = self.__fig.add_axes([0.3, 0.08, 0.12, 0.03])
        self.__ax_v2 = self.__fig.add_axes([0.45, 0.08, 0.12, 0.03]) # Adjusted spacing
        self.__txt_v1 = TextBox(self.__ax_v1, "v1", initial="0")
        self.__txt_v2 = TextBox(self.__ax_v2, "v2", initial="0")
        # On submit now only updates internal values, doesn't re-render
        self.__txt_v1.on_submit(self.__update_params_internal)
        self.__txt_v2.on_submit(self.__update_params_internal)

        # --- APPLY BUTTON (NEW) ---
        ax_apply = self.__fig.add_axes([0.60, 0.075, 0.1, 0.04])
        self.__btn_apply = Button(ax_apply, 'APPLY', color='#d4f1f9', hovercolor='#76c7c0')
        self.__btn_apply.on_clicked(self.__apply_changes)

        # --- CLASS BUTTONS ---
        self.__btns = {}
        for idx, name in CLASS_CONFIG.items():
            ax_b = self.__fig.add_axes([0.02, 0.9 - (idx*0.05), 0.12, 0.04])
            btn = Button(ax_b, name)
            btn.on_clicked(lambda e, i=idx: self.__switch_buffer(i))
            self.__btns[idx] = btn

        # --- NAVIGATION ---
        ax_prev = self.__fig.add_axes([0.18, 0.02, 0.1, 0.04])
        self.__btn_prev = Button(ax_prev, '<< Prev')
        self.__btn_prev.on_clicked(self.__on_prev)

        ax_next = self.__fig.add_axes([0.78, 0.02, 0.1, 0.04])
        self.__btn_next = Button(ax_next, 'Next >>')
        self.__btn_next.on_clicked(self.__on_next)

        # Initial Render
        self.__update_norm_ui_labels()
        self.__render_stencil()
        self.__refresh_view()

        # Connections
        self.__fig.canvas.mpl_connect('button_press_event', self.__on_press)
        self.__fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)
        self.__fig.canvas.mpl_connect('button_release_event', self.__on_release)
        self.__fig.canvas.mpl_connect('key_press_event', self.__on_key)

    def __set_norm_mode_ui(self, mode):
        """Update the selected mode in UI only"""
        self.norm_mode = mode
        self.__update_norm_ui_labels()
        # Visual feedback that mode is selected but not yet applied
        for name, btn in self.__norm_btns.items():
            btn.ax.set_facecolor('yellow' if name == self.norm_mode else '0.85')
        self.__fig.canvas.draw_idle()

    def __update_norm_ui_labels(self):
        """Updates text box labels and values to match current state"""
        p = self.__params[self.norm_mode]
        self.__txt_v1.label.set_text(p["l1"])
        self.__txt_v2.label.set_text(p["l2"])
        self.__txt_v1.set_val(str(p["v1"]))
        self.__txt_v2.set_val(str(p["v2"]))

    def __update_params_internal(self, text):
        """Store values from text boxes without rendering"""
        try:
            self.__params[self.norm_mode]["v1"] = float(self.__txt_v1.text)
            self.__params[self.norm_mode]["v2"] = float(self.__txt_v2.text)
        except ValueError: 
            pass

    def __apply_changes(self, event):
        """Triggered by Apply Button: recalculate and re-render"""
        # Ensure latest text box values are captured
        self.__update_params_internal(None)
        
        # Reset button colors to show 'Applied' state
        for name, btn in self.__norm_btns.items():
            btn.ax.set_facecolor('0.85')
            btn.label.set_color('red' if name == self.norm_mode else 'black')
        
        self.__render_stencil()
        self.__refresh_view()

    def __render_stencil(self):
        """Core Normalization Logic (GPU)"""
        data = self.__raw_gpu
        p = self.__params[self.norm_mode]
        v1, v2 = p["v1"], p["v2"]

        if self.norm_mode == "Linear":
            res = cp.clip(data, v1, v2)
            res = (res - v1) / (v2 - v1 + 1e-7)
        elif self.norm_mode == "Skewed":
            res = 1.0 / (1.0 + cp.exp(-(data - v1) / (v2 + 1e-7)))
        elif self.norm_mode == "Z-Score":
            mean, std = cp.mean(data), cp.std(data)
            res = (data - (mean + v2)) / (std + 1e-7)
            res = cp.clip(res, -v1, v1)
            res = (res + v1) / (2 * v1)
        elif self.norm_mode == "Log":
            res = cp.log1p(cp.maximum(0, data * v1 + v2))
            res = (res - res.min()) / (res.max() - res.min() + 1e-7)

        self.__stencil_cpu = cp.asnumpy((res * 255).astype(cp.uint8))

    def __refresh_view(self):
        self.__ax.clear()
        self.__ax.set_axis_off()
        self.__ax.imshow(self.__stencil_cpu, cmap='gray', vmin=0, vmax=255, zorder=1)
        
        mask = cp.asnumpy(self.__mask_buffers[:, :, self.__active_idx])
        cm = LinearSegmentedColormap.from_list('c', [(0,0,0,0), '#00FF00'], N=2)
        self.__ax.imshow(mask, cmap=cm, zorder=2, alpha=0.4)
        
        self.__line_marker, = self.__ax.plot([], [], color='#00FF00', lw=2, animated=True, zorder=10)
        self.__erase_marker = plt.Circle((0,0), self.__erase_r, color='red', fill=False, animated=True, zorder=11)
        self.__ax.add_patch(self.__erase_marker)
        
        self.__fig.canvas.draw()
        self.__bg_cache = self.__fig.canvas.copy_from_bbox(self.__ax.bbox)

    def __switch_buffer(self, idx):
        self.__active_idx = idx
        for i, btn in self.__btns.items():
            btn.label.set_color('red' if i == self.__active_idx else 'black')
        self.__refresh_view()

    # (Interaction logic remains same as provided in original)
    def __on_press(self, event):
        if event.inaxes == self.__ax:
            self.__active_button = event.button
            self.__last_coord = (event.ydata, event.xdata)
            self.__temp_pts = [self.__last_coord]

    def __on_motion(self, event):
        if self.__active_button and event.inaxes == self.__ax:
            curr = (event.ydata, event.xdata)
            self.__temp_pts.append(curr)
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

    def __on_prev(self, event):
        self.status = "prev"; plt.close(self.__fig)

    def __on_next(self, event):
        self.status = "next"; plt.close(self.__fig)

    def __on_key(self, event):
        if event.key == 'enter': self.__on_next(None)
        elif event.key == 'q': self.status = "quit"; plt.close(self.__fig)

    def show(self):
        plt.show()
        return self.status
