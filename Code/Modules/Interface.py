import tkinter as tk
from tkinter import ttk
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from .NormalizationEngine import NormalizationEngine
from .AnnotatorEngine import AnnotatorEngine

class AnnotatorUI:
    def __init__(self, root, processor, save_callback):
        self.root = root
        self.proc = processor
        self.save_callback = save_callback
        self.curr_idx = 0
        
        # UI Control Variables
        self.norm_mode = tk.StringVar(value="Linear")
        self.v1_var = tk.StringVar(value="1000")
        self.v2_var = tk.StringVar(value="3500")
        self.class_var = tk.IntVar(value=1)
        
        # State Variables
        self.drawing = False
        self.bg_cache = None
        self.temp_coords = []
        self.last_pos = None
        self.brush_size = 5
        self.eraser_radius = 20
        self.is_closed = False

        self.setup_ui()
        self.load_frame_engine()
        
        # Fix for the "Busy Terminal":
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

    def setup_ui(self):
        try: self.root.state('zoomed')
        except: self.root.attributes('-zoomed', True)
        self.root.configure(bg="#1a1a1a")

        # SIDEBAR
        sidebar = tk.Frame(self.root, bg="#252525", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="NORMALIZATION", fg="#00ffcc", bg="#252525", font=("Arial", 10, "bold")).pack(pady=10)
        ttk.Combobox(sidebar, textvariable=self.norm_mode, values=["Linear", "Skewed", "Z-Score", "Log"]).pack(pady=5)
        
        f = tk.Frame(sidebar, bg="#252525")
        f.pack(pady=5)
        tk.Label(f, text="V1:", fg="white", bg="#252525").grid(row=0, column=0)
        tk.Entry(f, textvariable=self.v1_var, width=8).grid(row=0, column=1)
        tk.Label(f, text="V2:", fg="white", bg="#252525").grid(row=1, column=0)
        tk.Entry(f, textvariable=self.v2_var, width=8).grid(row=1, column=1)
        
        tk.Button(sidebar, text="APPLY VIEW", bg="#444", fg="white", command=self.apply_normalization).pack(pady=10, fill=tk.X, padx=20)

        tk.Label(sidebar, text="CLASSES", fg="#00ffcc", bg="#252525", font=("Arial", 10, "bold")).pack(pady=10)
        classes = {1: "Bones", 2: "L_Pleural", 3: "R_Pleural", 4: "Mediastinum", 5: "Abdominal"}
        for val, name in classes.items():
            tk.Radiobutton(sidebar, text=name, variable=self.class_var, value=val, bg="#252525", fg="#aaa", selectcolor="black", command=self.refresh_plot).pack(anchor=tk.W, padx=30)

        # CANVAS
        self.fig, self.ax = plt.subplots(facecolor="#1a1a1a")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ARTISTS
        self.temp_line, = self.ax.plot([], [], lw=2, animated=True)
        self.eraser_cursor = Circle((0,0), self.eraser_radius, color='red', fill=False, lw=2, animated=True)
        self.ax.add_patch(self.eraser_cursor)
        self.eraser_cursor.set_visible(False)

        # FOOTER
        footer = tk.Frame(self.root, bg="#1a1a1a")
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(footer, text="PREV", width=10, command=self.prev_f).pack(side=tk.LEFT, padx=50, pady=10)
        tk.Button(footer, text="SAVE & EXIT", bg="#28a745", fg="white", font=("Arial", 11, "bold"), command=self.on_save_and_exit).pack(side=tk.LEFT, expand=True)
        tk.Button(footer, text="NEXT", width=10, command=self.next_f).pack(side=tk.RIGHT, padx=50, pady=10)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def load_frame_engine(self):
        print(f"[*] Displaying Frame {self.curr_idx + 1}")
        frame = self.proc.frames[self.curr_idx]
        self.engine = AnnotatorEngine(frame.pixel_data)
        self.apply_normalization()

    def apply_normalization(self):
        try:
            v1, v2 = float(self.v1_var.get()), float(self.v2_var.get())
            view_8bit = NormalizationEngine.run(self.engine.raw_gpu, self.norm_mode.get(), v1, v2)
            self.engine.update_stencil(view_8bit)
            self.refresh_plot()
        except Exception as e:
            print(f"[!] Norm Error: {e}")

    def refresh_plot(self):
        if self.engine.view_stencil_cpu is None: return
        self.ax.clear()
        self.ax.imshow(self.engine.view_stencil_cpu, cmap='gray')
        
        mask = self.engine.get_mask_cpu(self.class_var.get())
        cm = LinearSegmentedColormap.from_list('m', [(0,0,0,0), '#00ff00'], N=2)
        self.ax.imshow(mask, cmap=cm, alpha=0.3)
        
        self.temp_line, = self.ax.plot([], [], lw=2, animated=True)
        self.ax.add_patch(self.eraser_cursor)
        self.ax.axis('off')
        self.canvas.draw()
        self.bg_cache = self.canvas.copy_from_bbox(self.ax.bbox)

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.drawing = True
            self.last_button = event.button
            self.last_pos = (event.ydata, event.xdata)
            if event.button == 3:
                self.eraser_cursor.set_visible(True)
                self.temp_line.set_visible(False)
            else:
                self.temp_coords = [(event.xdata, event.ydata)]
                self.eraser_cursor.set_visible(False)
                self.temp_line.set_visible(True)
                self.temp_line.set_color('#00ff00')
            self.paint(event)

    def on_move(self, event):
        if self.drawing and event.inaxes == self.ax:
            curr_pos = (event.ydata, event.xdata)
            if self.last_button == 3:
                self.eraser_cursor.center = (event.xdata, event.ydata)
                self.engine.apply_brush(int(event.ydata), int(event.xdata), self.eraser_radius, self.class_var.get(), erase=True)
            else:
                dist_to_start = np.hypot(curr_pos[0] - self.temp_coords[0][1], curr_pos[1] - self.temp_coords[0][0])
                self.is_closed = dist_to_start < 15 and len(self.temp_coords) > 20
                self.temp_line.set_color('yellow' if self.is_closed else '#00ff00')
                self.engine.apply_stroke(self.last_pos[0], self.last_pos[1], curr_pos[0], curr_pos[1], self.brush_size, self.class_var.get())
                self.temp_coords.append((event.xdata, event.ydata))
                x, y = zip(*self.temp_coords)
                self.temp_line.set_data(x, y)

            if self.bg_cache:
                self.canvas.restore_region(self.bg_cache)
                self.ax.draw_artist(self.eraser_cursor if self.last_button == 3 else self.temp_line)
                self.canvas.blit(self.ax.bbox)
            self.last_pos = curr_pos

    def on_release(self, event):
        if self.drawing:
            self.drawing = False
            if self.last_button == 1 and self.is_closed:
                self.engine.fill_closed_curve(self.temp_coords, self.class_var.get())
            self.eraser_cursor.set_visible(False)
            self.refresh_plot()

    def paint(self, event):
        r = self.eraser_radius if self.last_button == 3 else self.brush_size
        self.engine.apply_brush(int(event.ydata), int(event.xdata), r, self.class_var.get(), (self.last_button==3))

    def on_save_and_exit(self):
        self.sync_mask()
        self.save_callback()
        self.cleanup_and_exit()

    def cleanup_and_exit(self):
        print("[*] Cleaning up GPU resources and exiting...")
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def sync_mask(self):
        for i in range(1, 6):
            self.proc.frames[self.curr_idx].masks[i] = self.engine.get_mask_cpu(i)

    def next_f(self):
        self.sync_mask()
        if self.curr_idx < len(self.proc.frames)-1:
            self.curr_idx += 1; self.load_frame_engine()

    def prev_f(self):
        self.sync_mask()
        if self.curr_idx > 0:
            self.curr_idx -= 1; self.load_frame_engine()

