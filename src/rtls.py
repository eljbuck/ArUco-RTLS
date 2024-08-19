import cv2
import numpy as np

# Set up camera calibration
camera_calibration_path = "../cameras/logitech_webcam/calib_data/MultiMatrix.npz"

camera_calibration_data = np.load(camera_calibration_path)

cam_matrix = camera_calibration_data["camMatrix"]
dist_coef = camera_calibration_data["distCoef"]
r_vector = camera_calibration_data["rVector"]
t_vector = camera_calibration_data["tVector"]

# Load Aruco dictionary
arudo_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Defined constants
MARKER_SIZE_CM = 1.918

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display frame
    cv2.imshow('RTLS', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
