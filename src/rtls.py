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

def compute_relative_distance(tvec_anchor, tvec_tag):
    # Compute relative translation vector
    tvec_relative = tvec_tag - tvec_anchor

    distance_from_camera = tvec_tag[0][2]

    # Extract x and y distances
    distance_x = tvec_relative[0, 0]
    distance_y = tvec_relative[0, 1]
    distance_z = max(0, tvec_anchor[0][2] - tvec_tag[0][2])  # Assume anchor is on ground

    return distance_x, distance_y, distance_z

def estimate_pose(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_CM, cameraMatrix=cam_matrix, distCoeffs=dist_coef)
        anchor_tvec = None
        ground_depth = None
        for i, id in enumerate(ids):
            if id == 1:  # Assume the anchor marker has ID 1
                anchor_rvec = rvec[i]
                anchor_tvec = tvec[i]
                break
        
        for i, id in enumerate(ids):
            if id != 1:  # For other markers
                marker_rvec = rvec[i]
                marker_tvec = tvec[i]

                if anchor_tvec is not None:
                    x, y, z = compute_relative_distance(anchor_tvec, marker_tvec)
                    print(f"Marker {id} coords (relative to anchor): ({x}, {y}, {z})")
                

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
