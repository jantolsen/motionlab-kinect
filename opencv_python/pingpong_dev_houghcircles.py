import cv2
import numpy as np

# Find HSV value range for desired object
class HSV_rangefinder():
    
    def __init__(self):
        cv2.namedWindow('HSV_rangefinder')
        cv2.createTrackbar('H - min', 'HSV_rangefinder', 0, 179, self.callback)
        cv2.createTrackbar('H - max', 'HSV_rangefinder', 179, 179, self.callback)
        cv2.createTrackbar('S - min', 'HSV_rangefinder', 0, 255, self.callback)
        cv2.createTrackbar('S - max', 'HSV_rangefinder', 255, 255, self.callback)
        cv2.createTrackbar('V - min', 'HSV_rangefinder', 0, 255, self.callback)
        cv2.createTrackbar('V - max', 'HSV_rangefinder', 255, 255, self.callback)
    
    # Dummy function to pass as function pointer to trackbar
    def callback(self, input):
        pass

    # Create a HSV mask on original input frame
    def hsv_mask_finder(self, frame):
        
        # Transform original frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        # Show HSV mask
        cv2.imshow('HSV_rangefinder', hsv_mask)

        # Return HSV range values
        return (hsv_lower, hsv_upper)

def main():
    
    # Initialize web camera object
    cam = cv2.VideoCapture(0)

    # Initialize HSV range finder
    HSV = HSV_rangefinder()

    # HSV range (pingpong ball) 
    # (found earlier from HSV_rangefinder)
    hsv_low = np.array([0,115,100])
    hsv_upp = np.array([30,255,255])

    # Loop
    while True:
        # Read frame from webcam
        ret, frame = cam.read()

        # Show webcam frame
        cv2.imshow('Original Frame', frame)

        # Convert original frame to HSV color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Start HSV range finder (function returns lower and upper range)
        # (_,_) = HSV.hsv_mask_finder(frame)

        hsv_mask = cv2.inRange(hsv, hsv_low, hsv_upp)
        cv2.imshow('Masked_hsv', hsv_mask)
        
        # cv2.imshow('test', hsv_mask[2,:])
        img_blur = cv2.medianBlur(hsv_mask,15)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gauss = cv2.GaussianBlur(gray,(5,5), 0, 0)
        

        ret, th = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('test_gray', img_gauss)
        cv2.imshow('test', th)


        circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, minDist = 100,
            param1=50, param2 = 30, minRadius = 20, maxRadius = 40)
        
        # circles = np.uint16(np.around(circles))
        if not circles is None:
            
            for i in circles[0,:]:
                cv2.circle(frame, (i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('circles', frame)
        # cimg = cv2.cvtColor(hsv_mask, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', cimg)
        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Test print
    print(circles)
    # Release web camera object
    cam.release()
    # Destroy all opencv windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()