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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Defined constants
MARKER_SIZE_CM = 1.918

# Initialize camera
cap = cv2.VideoCapture(0)

def estimate_pose(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_CM, cameraMatrix=cam_matrix, distCoeffs=dist_coef)
        
        # Example of using anchor marker
        for i, id in enumerate(ids):
            if id == 1:  # Assume the anchor marker has ID 1
                anchor_rvec = rvec[i]
                anchor_tvec = tvec[i]
                print("Anchor marker pose (relative to camera):", anchor_rvec, anchor_tvec)
                break
        
        for i, id in enumerate(ids):
            if id != 1:  # For other markers
                marker_rvec = rvec[i]
                marker_tvec = tvec[i]
                print(f"Marker {id} pose (relative to camera):", marker_rvec, marker_tvec)
                # Compute global position
                # Assuming `anchor_rvec` and `anchor_tvec` are known
                # Compute position relative to the anchor marker (0, 0, 0)
                # Apply transformation from camera coordinates to global coordinates
                # Implement transformation logic here

    return corners, ids

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate poses
    corners, ids = estimate_pose(frame)

    # Display frame
    cv2.imshow('RTLS', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
