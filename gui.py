import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class AppGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Measurement Tool")

        self.video_label = tk.Label(root)
        self.video_label.pack()


    def update_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(pil_image)

        self.video_label.configure(image=tk_image)
        self.video_label.imgtk = tk_image