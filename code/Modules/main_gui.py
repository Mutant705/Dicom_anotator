import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import cupy as cp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from engine import ImageEngine

class App(tk.Tk):
    def __init__(self, frame_obj):
        super().__init__()
        self.title("DICOM GPU Visualizer - Tkinter Edition")
        self.geometry("1400x900")
        
        # Initialize Logic Engine
        self.engine = ImageEngine(frame_obj)
        self.status = "stay"  # For navigation tracking
        
        # UI State Variables
        self.norm_mode = tk.StringVar(value="Linear")
        self.v1_var = tk.DoubleVar(value=1000.0)
        self.v2_var = tk.DoubleVar(value=3500.0)
        self.brush_size = 10
        
        self.__setup_ui()
        
        # Initial Draw
        self.__start_apply_thread()

    def __setup_ui(self):
        # Main Layout: Left Sidebar and Right Canvas
        self.sidebar = ttk.Frame(self, padding="10", width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # --- Normalization Section ---
        ttk.Label(self.sidebar, text="Normalization", font=('Arial', 12, 'bold')).pack(pady=10)
        
        modes = ["Linear", "Skewed", "Z-Score", "Log"]
        for mode in modes:
            ttk.Radiobutton(self.sidebar, text=mode, variable=self.norm_mode, 
                            value=mode, command=self.__on_mode_change).pack(anchor=tk.W)

        ttk.Label(self.sidebar, text="Param V1:").pack(pady=(10, 0))
        self.ent_v1 = ttk.Entry(self.sidebar, textvariable=self.v1_var)
        self.ent_v1.pack(fill=tk.X)

        ttk.Label(self.sidebar, text="Param V2:").pack(pady=(5, 0))
        self.ent_v2 = ttk.Entry(self.sidebar, textvariable=self.v2_var)
        self.ent_v2.pack(fill=tk.X)

        self.btn_apply = ttk.Button(self.sidebar, text="APPLY CHANGES", command=self.__start_apply_thread)
        self.btn_apply.pack(pady=15, fill=tk.X)

        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', pady=10)

        # --- Navigation Section ---
        ttk.Label(self.sidebar, text="Navigation", font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Button(self.sidebar, text="<< Previous Slice", command=self.__on_prev).pack(fill=tk.X, pady=2)
        ttk.Button(self.sidebar, text="Next Slice >>", command=self.__on_next).pack(fill=tk.X, pady=2)
        ttk.Button(self.sidebar, text="QUIT (Q)", command=self.__on_quit).pack(fill=tk.X, pady=20)

        # --- Matplotlib Area ---
        self.fig = Figure(figsize=(10, 8), facecolor='#2c2c2c')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Initialize plot handles
        blank = np.zeros((self.engine.h, self.engine.w))
        self.img_handle = self.ax.imshow(blank, cmap='gray', vmin=0, vmax=255)
        self.mask_handle = self.ax.imshow(blank, cmap=self.engine.mask_cm, alpha=0.4)

        # Connect Events for Drawing
        self.canvas.mpl_connect('button_press_event', self.__on_press)
        self.canvas.mpl_connect('motion_notify_event', self.__on_motion)
        self.canvas.mpl_connect('button_release_event', self.__on_release)
        self.bind("<Key>", self.__on_key)

    # --- Logic Methods ---

    def __on_mode_change(self):
        """Update entry boxes with default params for that mode (optional)"""
        # You could add logic here to auto-fill v1/v2 based on the mode selected
        pass

    def __start_apply_thread(self):
        self.btn_apply.config(state=tk.DISABLED, text="Processing...")
        
        # Background threading prevents UI lag during GPU math
        t = threading.Thread(target=self.__run_normalization)
        t.daemon = True
        t.start()

    def __run_normalization(self):
        mode = self.norm_mode.get()
        v1 = self.v1_var.get()
        v2 = self.v2_var.get()
        
        stencil = self.engine.normalize(mode, v1, v2)
        mask = self.engine.get_active_mask()
        
        # Schedule UI update on main thread
        self.after(0, self.__update_display, stencil, mask)

    def __update_display(self, stencil, mask):
        self.img_handle.set_data(stencil)
        self.mask_handle.set_data(mask)
        self.canvas.draw_idle()
        self.btn_apply.config(state=tk.NORMAL, text="APPLY CHANGES")

    # --- Interaction Methods (Mask Drawing) ---

    def __on_press(self, event):
        if event.inaxes == self.ax:
            self.drawing = True
            self.__paint(event)

    def __on_motion(self, event):
        if hasattr(self, 'drawing') and self.drawing and event.inaxes == self.ax:
            self.__paint(event)

    def __on_release(self, event):
        self.drawing = False
        # Update display after a stroke
        mask = self.engine.get_active_mask()
        self.mask_handle.set_data(mask)
        self.canvas.draw_idle()

    def __paint(self, event):
        x, y = int(event.xdata), int(event.ydata)
        # Fast GPU-based mask update
        # (Assuming you keep the __gpu_op logic or similar in engine.py)
        # For brevity, let's assume engine handles the stroke logic:
        self.__apply_brush_gpu(y, x, event.button == 3)

    def __apply_brush_gpu(self, y, x, erase):
        r = self.brush_size
        y_s, y_e = max(0, y-r), min(self.engine.h, y+r+1)
        x_s, x_e = max(0, x-r), min(self.engine.w, x+r+1)
        
        # This part interacts directly with the engine's GPU buffer
        # This is very fast even for high-res images
        Y, X = cp.ogrid[-r:r+1, -r:r+1]
        brush = (X**2 + Y**2 <= r**2).astype(cp.uint8)
        
        # Extract the slice of the brush that fits in bounds
        b_slice = brush[y_s-(y-r):y_s-(y-r)+(y_e-y_s), x_s-(x-r):x_s-(x-r)+(x_e-x_s)]
        
        if erase:
            self.engine.mask_buffers[y_s:y_e, x_s:x_e, self.engine.active_mask_idx] &= ~b_slice
        else:
            self.engine.mask_buffers[y_s:y_e, x_s:x_e, self.engine.active_mask_idx] |= b_slice

    # --- Navigation Callbacks ---

    def __on_prev(self):
        self.status = "prev"
        self.destroy()

    def __on_next(self):
        self.status = "next"
        self.destroy()

    def __on_quit(self):
        self.status = "quit"
        self.destroy()

    def __on_key(self, event):
        if event.char.lower() == 'q':
            self.__on_quit()
        elif event.keysym == 'Return':
            self.__start_apply_thread()
