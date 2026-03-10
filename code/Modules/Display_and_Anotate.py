"""
MODULE: Display_and_Anotate.py (Updated with Bresenham Line Interpolation)
TASKS DONE:
    - Fixed Lag: Optimized rendering by only updating active overlay.
    - Continuous Curves: Interpolates between mouse events to prevent "dotted" lines.
    - Right-click Drag: Smooth, connected 7x7 brush strokes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap


class DICOMVisualizer:
    def __init__(self, frame, window_start=0):
        self.__frame = frame
        self.__raw_data = frame.pixel_data.astype(np.int32)
        self.__window_val = window_start

        # Internal State
        self.__class_names = ["Tumor", "Tissue", "Bone"]
        self.__mask_3d = np.zeros(
            (*self.__raw_data.shape, len(self.__class_names)), dtype=np.uint8
        )
        self.__active_idx = 0
        self.__is_eraser = False
        self.__visible = [True] * len(self.__class_names)

        # Tracking for connected lines
        self.__is_drawing = False
        self.__last_x = None
        self.__last_y = None

        self.__setup_ui()

    def __setup_ui(self):
        self.__fig = plt.figure(figsize=(11, 9))
        self.__fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)
        self.__ax = self.__fig.add_axes([0.1, 0.15, 0.8, 0.75])

        display_buf = self.__calculate_display(self.__window_val)
        self.__img_handle = self.__ax.imshow(
            display_buf, cmap="gray", vmin=0, vmax=255, zorder=1
        )
        self.__ax.set_axis_off()

        self.__overlays = []
        marker_color = "#00FF00"
        for i in range(len(self.__class_names)):
            cmap = LinearSegmentedColormap.from_list(
                "ov", [(0, 0, 0, 0), marker_color], N=2
            )
            ov = self.__ax.imshow(
                np.zeros_like(self.__raw_data),
                cmap=cmap,
                alpha=0.8,
                vmin=0,
                vmax=1,
                zorder=2,
                interpolation="nearest",
            )
            self.__overlays.append(ov)

        # UI Widgets
        slice_max = max(0, self.__raw_data.max() - 255)
        ax_slider = self.__fig.add_axes([0.25, 0.05, 0.5, 0.03])
        start_pos = (
            0
            if str(self.__window_val).lower() == "normalize"
            else int(self.__window_val)
        )
        self.__slider = Slider(
            ax_slider, "Window", 0, slice_max, valinit=start_pos, valfmt="%d"
        )
        self.__slider.on_changed(self.__update_window)

        self.__btns = []
        w = 0.8 / len(self.__class_names)
        for i, name in enumerate(self.__class_names):
            btn_ax = self.__fig.add_axes([0.1 + (i * w), 0.91, w - 0.01, 0.05])
            b = Button(btn_ax, name)
            b.label.set_color("red" if i == self.__active_idx else "black")
            b.on_clicked(lambda e, idx=i: self.__handle_class_toggle(idx))
            self.__btns.append(b)

        er_ax = self.__fig.add_axes([0.02, 0.02, 0.03, 0.03])
        self.__btn_erase = Button(er_ax, "E", color="lightgray")
        self.__btn_erase.on_clicked(self.__toggle_eraser)

        # Event Connections
        self.__fig.canvas.mpl_connect("button_press_event", self.__on_press)
        self.__fig.canvas.mpl_connect("button_release_event", self.__on_release)
        self.__fig.canvas.mpl_connect("motion_notify_event", self.__on_motion)
        self.__fig.canvas.mpl_connect("key_press_event", self.__on_key)

    def __calculate_display(self, win_start):
        if str(win_start).lower() == "normalize":
            d_min, d_max = self.__raw_data.min(), self.__raw_data.max()
            if d_max == d_min:
                return np.zeros_like(self.__raw_data, dtype=np.uint8)
            return ((self.__raw_data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        ws = int(win_start)
        buf = np.zeros_like(self.__raw_data, dtype=np.uint8)
        mask = (self.__raw_data >= ws) & (self.__raw_data <= ws + 255)
        buf[mask] = (self.__raw_data[mask] - ws).astype(np.uint8)
        return buf

    def __update_window(self, val):
        self.__img_handle.set_data(self.__calculate_display(val))
        self.__fig.canvas.draw_idle()

    # --- ENHANCED DRAWING LOGIC ---

    def __on_press(self, event):
        if event.inaxes == self.__ax and event.button == 3:
            self.__is_drawing = True
            self.__last_x, self.__last_y = event.xdata, event.ydata
            self.__apply_brush(event.xdata, event.ydata)

    def __on_release(self, event):
        if event.button == 3:
            self.__is_drawing = False
            self.__last_x, self.__last_y = None, None

    def __on_motion(self, event):
        if self.__is_drawing and event.inaxes == self.__ax:
            # Connect the last recorded point to the current point
            self.__draw_line(self.__last_x, self.__last_y, event.xdata, event.ydata)
            self.__last_x, self.__last_y = event.xdata, event.ydata
            self.__fig.canvas.draw_idle()

    def __draw_line(self, x0, y0, x1, y1):
        """Line interpolation to ensure no gaps when moving mouse fast."""
        # Calculate distance between points
        dist = int(np.hypot(x1 - x0, y1 - y0))
        if dist < 1:
            self.__apply_brush(x1, y1)
            return

        # Linear interpolation between last point and current point
        for i in range(dist + 1):
            t = i / dist
            curr_x = x0 + t * (x1 - x0)
            curr_y = y0 + t * (y1 - y0)
            self.__apply_brush(curr_x, curr_y)

    def __apply_brush(self, x_f, y_f):
        if not self.__visible[self.__active_idx]:
            return

        x, y = int(round(x_f)), int(round(y_f))
        val = 0 if self.__is_eraser else 1

        r = 3
        y_grid, x_grid = np.ogrid[-r : r + 1, -r : r + 1]
        brush = (x_grid**2 + y_grid**2 <= r**2).astype(np.uint8)

        img_h, img_w = self.__raw_data.shape
        y_s, y_e = max(0, y - r), min(img_h, y + r + 1)
        x_s, x_e = max(0, x - r), min(img_w, x + r + 1)

        b_y_s = y_s - (y - r)
        b_y_e = b_y_s + (y_e - y_s)
        b_x_s = x_s - (x - r)
        b_x_e = b_x_s + (x_e - x_s)

        b_slice = brush[b_y_s:b_y_e, b_x_s:b_x_e]

        if val == 1:
            self.__mask_3d[y_s:y_e, x_s:x_e, self.__active_idx] |= b_slice
        else:
            self.__mask_3d[y_s:y_e, x_s:x_e, self.__active_idx] &= ~b_slice

        self.__overlays[self.__active_idx].set_data(
            self.__mask_3d[:, :, self.__active_idx]
        )

    # --- UI HELPERS ---

    def __handle_class_toggle(self, idx):
        if self.__active_idx == idx:
            self.__visible[idx] = not self.__visible[idx]
        else:
            self.__active_idx = idx
            self.__visible[idx] = True
        self.__overlays[idx].set_visible(self.__visible[idx])
        for i, b in enumerate(self.__btns):
            b.label.set_color(
                "red" if (i == self.__active_idx and self.__visible[i]) else "black"
            )
        self.__fig.canvas.draw_idle()

    def __on_key(self, event):
        if event.key == "q":
            self.__frame.pixel_data = self.__mask_3d
            plt.close(self.__fig)

    def __toggle_eraser(self, event):
        self.__is_eraser = not self.__is_eraser
        self.__btn_erase.color = "salmon" if self.__is_eraser else "lightgray"
        self.__fig.canvas.draw_idle()

    def show(self):
        plt.show()
