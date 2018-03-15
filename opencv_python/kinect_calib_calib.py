import cv2
import numpy as np
import os
from glob import glob
import shelve

isRGB = False

rgbFolder = 'C:/Users/Workstation/Documents/Kinect Calibration/RGB_Frame/'
irFolder = 'C:/Users/Workstation/Documents/Kinect Calibration/IR_Frame/'

rgbCameraPath = 'C:/Users/Workstation/Documents/Kinect Calibration/results/RGB'
irCameraPath = 'C:/Users/Workstation/Documents/Kinect Calibration/results/IR'

pattern_size = (6,9)
square_size = 0.035 # metres
if (isRGB):
    img_names = glob(rgbFolder + '*.jpg')
    CAMERA_PATH = rgbCameraPath
else:
    img_names = glob(irFolder + '*.jpg')
    CAMERA_PATH = irCameraPath

# Create object points
pattern_points = np.zeros((np.prod(pattern_size),3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1,2)
pattern_points *= square_size

obj_points = []
img_points = []

h, w = 0, 0

for fn in img_names:
    print('processing %s ...' % fn)

    img = cv2.imread(fn, 0)
    if img is None:
        print('failed to load', fn)
        continue
    
    h,w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, pattern_size, flags = cv2.CALIB_CB_ADAPTIVE_THRESH)

    if found:
        cv2.cornerSubPix(img, corners, (5,5), (-1,1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if not found:
        print('Chessboard not found')
        continue

    img_points.append(corners.reshape(-1,2))
    obj_points.append(pattern_points)

    # save img_points for future stereo calibration
    img_file = shelve.open(os.path.splitext(fn)[0],'n')
    img_file['img_points'] = corners.reshape(-1,2)
    img_file.close()

    print('ok')

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                                   img_points,
                                                                   (w,h),
                                                                   None,
                                                                   None,
                                                                   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.001),
                                                                   flags = 0)

#save calibration results
camera_file = shelve.open(CAMERA_PATH, 'n')
camera_file['camera_matrix'] = camera_matrix
camera_file['dist_coefs'] = dist_coefs
camera_file.close()

print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

print('done')