import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class AppGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Measurement Tool")

        self.var_crop = tk.DoubleVar(value=0.2)
        self.setup_ui()

    def setup_ui(self):
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.control_frame = tk.LabelFrame(self.root, text="Controls", padx=10, pady=10)
        self.control_frame.pack(side="left", padx=5)

        self.scale_crop = tk.Scale(
            self.control_frame,
            from_=0.0,
            to=0.45,
            resolution=0.01,
            orient="horizontal",
            variable=self.var_crop,
            length=200
        )
        self.scale_crop.pack(side="left", padx=5)

    def update_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(pil_image)

        self.video_label.configure(image=tk_image)
        self.video_label.imgtk = tk_image