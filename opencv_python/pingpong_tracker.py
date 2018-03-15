import cv2
import numpy as np

def callback(input):
    pass

cam = cv2.VideoCapture(0)

cv2.namedWindow('HSV_rangefinder')
cv2.createTrackbar('H - min', 'HSV_rangefinder', 0, 179, callback)
cv2.createTrackbar('H - max', 'HSV_rangefinder', 179, 179, callback)
cv2.createTrackbar('S - min', 'HSV_rangefinder', 0, 255, callback)
cv2.createTrackbar('S - max', 'HSV_rangefinder', 255, 255, callback)
cv2.createTrackbar('V - min', 'HSV_rangefinder', 0, 255, callback)
cv2.createTrackbar('V - max', 'HSV_rangefinder', 255, 255, callback)

while True:
    ret, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # cv2.imshow('HSV_rangefinder', hsv)

    # Get current position-values of the HSV-trackbars
    h_min = cv2.getTrackbarPos('H - min', 'HSV_rangefinder')
    h_max = cv2.getTrackbarPos('H - max', 'HSV_rangefinder')
    s_min = cv2.getTrackbarPos('S - min', 'HSV_rangefinder')
    s_max = cv2.getTrackbarPos('S - max', 'HSV_rangefinder')
    v_min = cv2.getTrackbarPos('V - min', 'HSV_rangefinder')
    v_max = cv2.getTrackbarPos('V - max', 'HSV_rangefinder')

    # Store trackbar values in upper and lower range arrays
    hsv_lower = np.array([h_min, s_min, v_min])
    hsv_upper = np.array([h_max, s_max, v_max])

    # HSV mask
    hsv_mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    cv2.imshow('HSV_rangefinder', hsv_mask)

    # Break loop on keystroke ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()