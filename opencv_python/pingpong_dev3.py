import cv2
import numpy as np
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

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

def midpoint(ptA, ptB):
    	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def main():
    
    # Initialize web camera object
    cam = cv2.VideoCapture(0)

    # Initialize HSV range finder
    HSV = HSV_rangefinder()

    # HSV range (pingpong ball) 
    # (found earlier from HSV_rangefinder)
    hsv_low = np.array([0,115,100])
    hsv_upp = np.array([30,255,255])

    # Detection variables
    # Morphology kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    # Drawing colors
    green_color = (0,255,0)    # BGR
    red_color = (0, 0, 255)    # BGR

    # Object radius range
    r_min = 10     # Minimum radius of circle-object (Pixel)
    r_max = 50     # Maximum radius of circle-object (Pixel)

    # Initializing object position of enclosing circle
    x = None       # x-position of enclosing circle
    y = None       # y-position of enclosing circle
    r = None       # radius of enclosing circle

    # Center coodinates
    cx = None  # x-position of center
    cy = None  # y-position of center

    filter_debug = True

    # Loop
    while True:
        # Read frame from webcam
        ret, frame = cam.read()

        # Show webcam frame
        # cv2.imshow('Original Frame', frame)

        # Convert original frame to HSV color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply threshold mask to HSV frame 
        # (Using upper and lower range from from HSV_MaskRange)
        hsv_thresh = cv2.inRange(hsv, hsv_low, hsv_upp)

        # Applying a median filter
        median_blur = cv2.medianBlur(hsv_thresh, 3)

        # Morphology (open = erosion followed by dilation)
        hsv_open = cv2.morphologyEx(hsv_thresh, cv2.MORPH_OPEN, 
                                    kernel, iterations = 2)
        # Frame of object detection
        object_mask = hsv_open

        # Hough Circles
        # --------------------------------------------------------------------------
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img_gauss = cv2.GaussianBlur(gray,(5,5), 0, 0)

        # circles = cv2.HoughCircles(img_gauss, cv2.HOUGH_GRADIENT, 1, minDist = 100,
        #     param1=50, param2 = 50, minRadius = 10, maxRadius = 0)
        
        # # circles = np.uint16(np.around(circles))
        # if not circles is None:
            
        #     for i in circles[0,:]:
        #         cv2.circle(frame, (i[0],i[1]),i[2],(0,255,0),2)
        #         cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

        # -------------------------------------------------------------------------------

        if filter_debug:
            cv2.imshow('HSV Frame', hsv)
            # cv2.imshow('HSV Thresh', hsv_thresh)
            # cv2.imshow('Medianblur', median_blur)
            # cv2.imshow('Morphology', hsv_open)
            # cv2.imshow('Gauss', img_gauss)
            # cv2.imshow('Hough Circles', frame)

        object_mask = cv2.resize(object_mask, (512, 424))
        frame = cv2.resize(frame, (512, 424))
        # Detection
        # -------------------------------------------------------------------------
        # Finding the contours in the mask
        (_,conts,_) = cv2.findContours(object_mask.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

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

             # Proceed if radius of object meets mininum size
            if r_min < r < r_max:
                # Draw enclosing circle
                cv2.circle(frame, (int(x), int(y)), int(r),
                            green_color, 2)

                # Draw center of object
                cv2.circle(frame, (cx, cy), 3, red_color, -1)

            

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 255), 2)

            box = perspective.order_points(box)
            
	        # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(frame, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        # draw the object sizes on the image
        cv2.putText(frame, "{:.1f}px".format(dA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(frame, "{:.1f}px".format(dB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(frame, "{:.1f}px".format(r*2),
            (int(trbrX + 100), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.imshow('Object Detection', frame)
        # -----------------------------------------------------------------------------

        (cnt,hir,k) = cv2.findContours(object_mask.copy(), 1, 2)
        x_rec,y_rec,w_rec,h_rec = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x_rec,y_rec),(x_rec+w_rec,y_rec+h_rec),(255,0,0),2)

        
        cv2.imshow('Test', frame)
        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Test print
    print(x,y,r)
    print(frame.shape)
    print(x_rec,y_rec,w_rec,h_rec)


    # Release web camera object
    cam.release()
    # Destroy all opencv windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()