import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import cv2

class AppGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Measurement Tool")

        #region - Class variables
        self.scale_crop = None
        self.btn_toggle_cam = None
        self.video_frame = None
        self.video_label = None
        self.var_stab_enable = tk.BooleanVar(value=True)
        self.var_stab_depth = tk.IntVar(value=15)
        self.var_camera_index = tk.IntVar(value=0)
        self.var_crop = tk.DoubleVar(value=0.2)
        self.cbk_toggle_cam = None
        #endregion

        self.setup_ui()

    def setup_ui(self):
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        container_control = tk.Frame(self.root, padx=10, pady=10)
        container_control.pack(fill="x", side="bottom")

        #region - Camera Controls
        container_camera = tk.LabelFrame(container_control, text="Camera Controls", padx=5, pady=5)
        container_camera.pack(side="left", padx=5, fill="y")

        tk.Label(container_camera, text="ID:").pack(side="left")
        ttk.Spinbox(container_camera, from_=0, to=10, width=3, textvariable=self.var_camera_index).pack(side="left", padx=5)

        self.btn_toggle_cam = ttk.Button(container_camera, text="Start Camera", command=self.on_toggle_click)
        self.btn_toggle_cam.pack(side="left", padx=5)
        #endregion

        #region - Crop (Zoom) Controls
        container_crop = tk.LabelFrame(container_control, text="Digital Zoom", padx=5, pady=5)
        container_crop.pack(side="left", padx=5, fill="y")

        self.scale_crop = tk.Scale(
            container_crop,
            from_=0.0,
            to=0.45,
            resolution=0.01,
            orient="horizontal",
            variable=self.var_crop,
            length=200
        )
        self.scale_crop.pack(side="left")
        #endregion

        #region - Stabilisation Controls
        container_stabilisation = tk.LabelFrame(container_control, text="Stabilisation", padx=5, pady=5)
        container_stabilisation.pack(side="left", padx=5, fill="y")

        chk_stabilisation = ttk.Checkbutton(container_stabilisation, text="Active", variable=self.var_stab_enable)
        chk_stabilisation.pack(side="left", padx=5)

        tk.Label(container_stabilisation,text="FramesNo: ").pack(side="left", padx=(10, 2))
        spin_frame = ttk.Spinbox(container_stabilisation, from_=1, to=50, width=3, textvariable=self.var_stab_depth)
        spin_frame.pack(side="left")
        #endregion

    def update_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(pil_image)

        self.video_label.configure(image=tk_image)
        self.video_label.imgtk = tk_image

    def on_toggle_click(self):
        if self.cbk_toggle_cam:
            self.cbk_toggle_cam()

    def clear_image(self):
        black_frame = np.zeros((729, 2565, 3), dtype=np.uint8)
        self.update_image(black_frame)

    def set_button_state(self, is_running):
        self.btn_toggle_cam.config(text="Stop Camera" if is_running else "Start Camera")