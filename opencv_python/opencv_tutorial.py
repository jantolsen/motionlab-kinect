import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50,100],[80,100], 'm', linewidth=2)
# plt.show()

# cv2.imwrite('new_image.png',img)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray frame',gray)
    
    equ = cv2.equalizeHist(gray)
    cv2.imshow('color frame',equ)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(gray.shape)
cap.release()

cv2.destroyAllWindows()
