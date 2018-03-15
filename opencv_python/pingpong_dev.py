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
        # cv2.imshow('Original Frame', frame)

        # Convert original frame to HSV color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV', hsv)
        

        # Start HSV range finder (function returns lower and upper range)
        # (_,_) = HSV.hsv_mask_finder(frame)

        hsv_mask = cv2.inRange(hsv, hsv_low, hsv_upp)
        cv2.imshow('Masked_hsv', hsv_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # hsv_open = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, 
        #                             kernel, iterations = 2)
        # cv2.imshow('Mask_open', hsv_open)
        
        img_blur = cv2.medianBlur(hsv_mask,5)
        cv2.imshow('media_blur', img_blur)

        # avg_blur = cv2.blur(hsv_mask,(5,5))
        # cv2.imshow('avg_blur', avg_blur)

        # gauss_blur = cv2.GaussianBlur(hsv_mask, (5,5), 0)
        # cv2.imshow('Gauss_blur', gauss_blur)

        dilate = cv2.dilate(img_blur, kernel, iterations=3)
        cv2.imshow('HSV_dilate', img_blur)

        # hsv_close = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, 
        #                             kernel, iterations = 3)
        # cv2.imshow('Mask_close', hsv_close)


        hsv_open = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, 
                                    kernel, iterations = 2)
        cv2.imshow('Mask_open', hsv_open)
        
        (_,conts,_) = cv2.findContours(hsv_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, cnts, -1, (0,0,255), 2)
        # cv2.imshow('Original Frame', frame)

        # Proceed if at least one contour was found
        if len(conts) > 0:
            # Find the largest contour
            c = max(conts, key = cv2.contourArea)

            # Compute the minimum enclosing circle 
            # (x-position: x, y-position: y, radius: r)
            ((x, y), r) = cv2.minEnclosingCircle(c)

            # Find the moments of the object
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx,cy)

            # Proceed if radius of object meets mininum size
            if 15 < r < 50:
                # Draw enclosing circle
                cv2.circle(hsv_open, (int(x), int(y)), int(r),
                            (0, 255, 0), 1)

                # Draw center of object
                cv2.circle(hsv_open, center, 3, (0,0,255), -1)

        cv2.imshow('Original Frame', hsv_open)
            
        # ret, otsu_thresh = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow('HSV_Otsu', otsu_thresh)

        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Test print
    print(x,y)
    print(frame.shape)


    # Release web camera object
    cam.release()
    # Destroy all opencv windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()