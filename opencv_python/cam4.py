import cv2
import numpy as np

def callback(x):
    pass

def main():
    
    cam = cv2.VideoCapture(0)

    hsv_thresh = np.zeros((300,512,1), np.uint8)
    cv2.namedWindow('hsv_threshold')

    cv2.createTrackbar('H_min','hsv_threshold',0,179,callback)
    cv2.createTrackbar('H_max','hsv_threshold',0,179,callback)
    cv2.createTrackbar('S_min','hsv_threshold',0,255,callback)
    cv2.createTrackbar('S_max','hsv_threshold',0,255,callback)
    cv2.createTrackbar('V_min','hsv_threshold',0,255,callback)
    cv2.createTrackbar('V_max','hsv_threshold',0,255,callback)
    cv2.createTrackbar('Erod', 'hsv_threshold',0,10,callback)
    cv2.createTrackbar('Dilate','hsv_threshold',0,10, callback)

    while True:
        ret, frame = cam.read()
        cv2.imshow('Original Frame', frame)

        blur = cv2.GaussianBlur(frame, (11,11),0)
        # cv2.imshow('Blur',blur)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV', hsv)

        # mask = cv2.inRange(hsv, greenLower, greenUpper)
	    # mask = cv2.erode(mask, None, iterations=2)
	    # mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow('hsv_threshold', hsv_thresh)

        # get current positions of three trackbars
        h_min = cv2.getTrackbarPos('H_min','hsv_threshold')
        h_max = cv2.getTrackbarPos('H_max','hsv_threshold')
        s_min = cv2.getTrackbarPos('S_min','hsv_threshold')
        s_max = cv2.getTrackbarPos('S_max','hsv_threshold')
        v_min = cv2.getTrackbarPos('V_min','hsv_threshold')
        v_max = cv2.getTrackbarPos('V_max','hsv_threshold')
        erod_i = cv2.getTrackbarPos('Erod','hsv_threshold')
        dila_i = cv2.getTrackbarPos('Dilate','hsv_threshold')

        hsv_lower = np.array([h_min, s_min, v_min])
        hsv_upper = np.array([h_max, s_max, v_max])

        findmask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        cv2.imshow('Find_mask_hsv', findmask)

        # HSV range pingpong
        # hsv_low = np.array([0,115,170])
        # hsv_upp = np.array([30,255,255])

        hsv_low = np.array([0,115,100])
        hsv_upp = np.array([30,255,255])

        # # HSV range paper test
        # hsv_low = np.array([64, 75, 55])
        # hsv_upp = np.array([179,255,175])
        
        
        mask = cv2.inRange(hsv, hsv_low, hsv_upp)
        cv2.imshow('Masked_hsv', mask)

        m_erode = cv2.erode(mask, None, iterations = 2)
        cv2.imshow('eroded_hsv', m_erode)

        m_dila = cv2.dilate(m_erode, None, iterations =  2)
        cv2.imshow('dilated_hsv', m_dila)

        # find contours in the mask and initialize the current
	    # (x, y) center of the ball
	    
        cnts = cv2.findContours(m_dila.copy(), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        # cnts = cv2.findContours(m_dila.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

	    # only proceed if at least one contour was found
        if len(cnts) > 0:
		    # find the largest contour in the mask, then use
		    # it to compute the minimum enclosing circle and
		    # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		    # only proceed if the radius meets a minimum size
            if radius > 10:
		    	# draw the circle and centroid on the frame,
		    	# then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
		    		(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        

        cv2.imshow('Original Frame', frame)
        # hsv[:,:] = [h;s;v]
        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()