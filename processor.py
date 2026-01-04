import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, aruco_type=cv2.aruco.DICT_5X5_1000, real_width=5.0):
        self.threshold_min = 50
        self.threshold_max = 150
        self.blur_kernel = (7, 7)
        self.morph_kernel = np.ones((5, 5), np.uint8)

    def process_frame(self, frame):
        input_frame = frame.copy()

        gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edged = cv2.Canny(blurred, self.threshold_min, self.threshold_max)

        dilated = cv2.dilate(edged, self.morph_kernel, iterations=1)
        cleaned = cv2.erode(dilated, self.morph_kernel, iterations=1)

        result_visual = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        return result_visual

