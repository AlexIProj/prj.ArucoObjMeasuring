import cv2
import numpy as np
from collections import deque

class ImageProcessor:
    def __init__(self, aruco_type=cv2.aruco.DICT_5X5_1000, real_width=5.0):
        #region - Image Processing Parameters
        self.threshold_min = 30
        self.threshold_max = 100
        self.blur_kernel = (7, 7)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        #endregion

        #region - ArUco parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.marker_real_width_cm = real_width
        self.pixel_per_cm = 0
        #endregion

        self.rect_history = deque(maxlen=15)

    def process_frame(self, frame, crop_margin=0.2):
        frame = self.apply_crop(frame, margin_percentage=crop_margin)

        input_frame = frame.copy()

        gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        #region - ArUco Detection
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(input_frame, corners, ids)

            aruco_perimeter_px = cv2.arcLength(corners[0], True)
            avg_side_px = aruco_perimeter_px / 4.0
            self.pixel_per_cm = avg_side_px / self.marker_real_width_cm

            #region - Debug ArUco
            text = f"Scara: {self.pixel_per_cm:.1f} px/cm"
            cv2.putText(input_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #endregion
        else:
            self.pixel_per_cm = 0
            cv2.putText(input_frame, "ArUco missing", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #endregion

        #region - Image Processing
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        edged = cv2.Canny(blurred, self.threshold_min, self.threshold_max)

        dilated = cv2.dilate(edged, self.morph_kernel, iterations=5)
        cleaned = cv2.erode(dilated, self.morph_kernel, iterations=5)
        #endregion

        #region - Detection and Measurement
        current_rect = None
        if self.pixel_per_cm > 0:
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                if cv2.contourArea(cnt) < 2000:
                    continue
                current_rect = cv2.minAreaRect(cnt)
                break
        if current_rect is not None:
            self.rect_history.append(current_rect)
        else:
            if len(self.rect_history) > 0:
                self.rect_history.popleft()
        if len(self.rect_history) > 0:
            all_centers = [r[0] for r in self.rect_history]
            all_size = [r[1] for r in self.rect_history]
            all_angles = [r[2] for r in self.rect_history]

            avg_center = np.mean(all_centers, axis=0)
            avg_size = np.mean(all_size, axis=0)
            avg_angle = np.mean(all_angles)

            raw_width = avg_size[0] / self.pixel_per_cm
            raw_height = avg_size[1] / self.pixel_per_cm

            offset_cm = 0.1

            width_cm = max(0, raw_width - offset_cm)
            height_cm = max(0, raw_height - offset_cm)

            avg_rect = (tuple(avg_center), tuple(avg_size), avg_angle)
            box = cv2.boxPoints(avg_rect)
            box = np.int32(box)

            cv2.drawContours(input_frame, [box], 0, (0, 255, 0), 2)
            text_dim = f"{width_cm:.1f}x{height_cm:.1f}cm"
            cv2.putText(input_frame, text_dim, (int(avg_center[0]), int(avg_center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #endregion

        edged_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        separator = np.ones((input_frame.shape[0], 5, 3), dtype=np.uint8) * 255
        combined_view =np.hstack((input_frame, separator, edged_bgr))

        return combined_view

    @staticmethod
    def apply_crop(frame, margin_percentage):
        if margin_percentage > 0.45:
            return frame
        original_h, original_w = frame.shape[:2]

        h, w, _ = frame.shape
        start_row = int(h * margin_percentage)
        end_row = int(h * (1.0 - margin_percentage))
        start_col = int(w * margin_percentage)
        end_col = int(w * (1.0 - margin_percentage))

        cropped_frame = frame[start_row:end_row, start_col:end_col]
        resized_frame = cv2.resize(cropped_frame, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        return resized_frame