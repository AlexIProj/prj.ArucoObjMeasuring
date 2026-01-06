import tkinter as tk
import cv2
from processor import ImageProcessor
from gui import AppGui

class MainApp:
    def __init__(self, root):
        self.root = root
        self.processor = ImageProcessor()
        self.gui = AppGui(root)
        self.gui.cbk_toggle_cam = self.toggle_camera

        self.cap = None
        self.is_running = False

        self.gui.clear_image()

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        camera_index = self.gui.var_camera_index.get()

        try:
            self.cap = cv2.VideoCapture(camera_index)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not self.cap.isOpened():
                print(f"Camera ID {camera_index} not working")
                return

            self.is_running = True

            self.gui.set_button_state(True)
            self.video_loop()

        except Exception as e:
            print(f"Camera error: {e}")

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

        self.gui.clear_image()
        self.gui.set_button_state(False)

    def video_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            current_crop = self.gui.var_crop.get()

            stab_enabled = self.gui.var_stab_enable.get()
            stab_depth = self.gui.var_stab_depth.get()

            show_measure = self.gui.var_show_measure.get()
            processed_frame = self.processor.process_frame(frame,
                                                           crop_margin=current_crop,
                                                           enable_stab=stab_enabled,
                                                           stab_depth=stab_depth,
                                                           draw_measurement=show_measure
                                                           )
            self.gui.update_image(processed_frame)
        else:
            self.stop_camera()
            return

        self.root.after(10, self.video_loop)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()