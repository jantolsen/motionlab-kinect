import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()

    img_blur = cv2.medianBlur(img,25)
    cimg = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2 = 30, minRadius = 10, maxRadius = 100)

    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        cv2.circle(cimg, (i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('Blurred', img_blur)
    cv2.imshow('Detected Circles',cimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()