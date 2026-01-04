import tkinter as tk
import cv2
from processor import ImageProcessor
from gui import AppGui

class MainApp:
    def __init__(self, root):
        self.root = root
        self.processor = ImageProcessor()
        self.gui = AppGui(root)

        self.cap = cv2.VideoCapture(0)
        self.video_loop()

    def video_loop(self):
        ret, frame = self.cap.read()

        if ret:
            current_crop = self.gui.scale_crop.get()
            processed_frame = self.processor.process_frame(frame, crop_margin=current_crop)
            self.gui.update_image(processed_frame)

        self.root.after(10, self.video_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()