import cv2
import numpy as np

# Find HSV value range for desired object
class HSV_MaskRange():
    
    def __init__(self):
        cv2.namedWindow('HSV_MaskRange')
        cv2.createTrackbar('H - min', 'HSV_MaskRange', 0, 179, self.callback)
        cv2.createTrackbar('H - max', 'HSV_MaskRange', 179, 179, self.callback)
        cv2.createTrackbar('S - min', 'HSV_MaskRange', 0, 255, self.callback)
        cv2.createTrackbar('S - max', 'HSV_MaskRange', 255, 255, self.callback)
        cv2.createTrackbar('V - min', 'HSV_MaskRange', 0, 255, self.callback)
        cv2.createTrackbar('V - max', 'HSV_MaskRange', 255, 255, self.callback)
    
    # Dummy function to pass as function pointer to trackbar
    def callback(self, input):
        pass
    # Create a HSV mask on original input frame
    def mask_finder(self, frame):
        
        # Transform original frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current position-values of the HSV-trackbars
        h_min = cv2.getTrackbarPos('H - min', 'HSV_MaskRange')
        h_max = cv2.getTrackbarPos('H - max', 'HSV_MaskRange')
        s_min = cv2.getTrackbarPos('S - min', 'HSV_MaskRange')
        s_max = cv2.getTrackbarPos('S - max', 'HSV_MaskRange')
        v_min = cv2.getTrackbarPos('V - min', 'HSV_MaskRange')
        v_max = cv2.getTrackbarPos('V - max', 'HSV_MaskRange')

        # Store trackbar values in upper and lower range arrays
        hsv_lower = np.array([h_min, s_min, v_min])
        hsv_upper = np.array([h_max, s_max, v_max])

        # Apply threshold to HSV frame
        hsv_mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)

        # Show HSV mask
        cv2.imshow('HSV_MaskRange', hsv_mask)

        # Return HSV range values
        return (hsv_lower, hsv_upper)

class ObjectDetection():
    
    def __init__(self):
        # Morphology kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        # Drawing colors
        self.green_color = (0,255,0)    # BGR
        self.red_color = (0, 0, 255)    # BGR

        # Object radius range
        self.r_min = 15     # Minimum radius of circle-object (Pixel)
        self.r_max = 50     # Maximum radius of circle-object (Pixel)

        # Initializing object position of enclosing circle
        self.x = None       # x-position of enclosing circle
        self.y = None       # y-position of enclosing circle
        self.r = None       # radius of enclosing circle

        # Center coodinates
        self.cx = None  # x-position of center
        self.cy = None  # y-position of center

    def detect(self, frame, HSV_lower, HSV_upper, filter_debug):
        # Convert original frame to HSV color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply threshold mask to HSV frame 
        # (Using upper and lower range from from HSV_MaskRange)
        hsv_thresh = cv2.inRange(hsv, HSV_lower, HSV_upper)

        # Applying a median filter
        median_blur = cv2.medianBlur(hsv_thresh, 3)

        # Morphology (open = erosion followed by dilation)
        hsv_open = cv2.morphologyEx(hsv_thresh, cv2.MORPH_OPEN, 
                                    self.kernel, iterations = 2)
        # Frame of object detection
        object_mask = hsv_open

        if filter_debug:
            cv2.imshow('HSV Frame', hsv)
            cv2.imshow('HSV Thresh', hsv_thresh)
            cv2.imshow('Medianblur', median_blur)
            cv2.imshow('Morphology', hsv_open)
            cv2.imshow('Hough Circles', frame)

        # Finding the contours in the mask
        (_,conts,_) = cv2.findContours(object_mask.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Proceed if at least one contour was found
        if len(conts) > 0:
            # Find the largest contour
            c = max(conts, key = cv2.contourArea)
        
            # Compute the minimum enclosing circle 
            # (x-position: x, y-position: y, radius: r)
            ((self.x, self.y), self.r) = cv2.minEnclosingCircle(c)

            # Find the moments of the object
            M = cv2.moments(c)
            self.cx = int(M["m10"] / M["m00"])
            self.cy = int(M["m01"] / M["m00"])

             # Proceed if radius of object meets mininum size
            if self.r_min < self.r < self.r_max:
                # Draw enclosing circle
                cv2.circle(frame, (int(self.x), int(self.y)), int(self.r),
                            self.green_color, 1)

                # Draw center of object
                cv2.circle(frame, (self.cx, self.cy), 3, self.red_color, -1)
        
        cv2.imshow('Object Detection', frame)
        cv2.imshow('Object Mask', object_mask)
        return (self.x, self.y, self.r)
    
def main():
    
    # Initialize web camera object
    cam = cv2.VideoCapture(0)

    # Initialize HSV range finder
    HSV = HSV_MaskRange()

    # HSV range (pingpong ball) 
    # (found earlier from HSV_MaskRange)
    hsv_low = np.array([0,115,100])
    hsv_upp = np.array([30,255,255])

    # hsv_low = np.array([0,0,250])
    # hsv_upp = np.array([172,255,255])

    # Initialize object detection
    OB = ObjectDetection()

    # Loop
    while True:
        # Read frame from webcam
        ret, frame = cam.read()

        # Show webcam frame
        # cv2.imshow('Original Frame', frame)

        # Start HSV range finder (function returns lower and upper range)
        (_,_) = HSV.mask_finder(frame)

        # Start Object detection 
        # (function returns object center coordinates and radius)
        (_,_,_) = OB.detect(frame, hsv_low, hsv_upp, False)

    # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release web camera object
    cam.release()
    # Destroy all opencv windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()