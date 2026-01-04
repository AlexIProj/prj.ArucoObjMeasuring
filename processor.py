import cv2
import numpy as np

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

    def process_frame(self, frame):
        #region - Digital Zoom (Crop)
        h, w, _ = frame.shape
        start_row = int(h * 0.20)
        end_row = int(h * 0.80)
        start_col = int(w * 0.20)
        end_col = int(w * 0.80)

        frame = frame[start_row:end_row, start_col:end_col]
        #endregion

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
        if self.pixel_per_cm > 0:
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                if cv2.contourArea(cnt) < 2000:
                    continue
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                offset_cm = 0.1
                raw_width = w / self.pixel_per_cm
                raw_height = h / self.pixel_per_cm

                width_cm = max(0, raw_width - offset_cm)
                height_cm = max(0, raw_height - offset_cm)

                box = cv2.boxPoints(rect)
                box = np.int32(box)

                cv2.drawContours(input_frame, [box], 0, (0, 255, 0), 2)
                text_dim = f"{width_cm:.1f}x{height_cm:.1f}cm"
                cv2.putText(input_frame, text_dim, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #endregion

        edged_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        separator = np.ones((input_frame.shape[0], 5, 3), dtype=np.uint8) * 255
        combined_view =np.hstack((input_frame, separator, edged_bgr))

        return combined_view