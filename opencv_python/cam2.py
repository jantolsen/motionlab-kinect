import cv2
import sys
import numpy as np


# Set up tracker
def tracker_type(name):
    if name == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
        print(tracker_type)
    if name == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if name == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if name == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if name == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if name == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    return tracker

# Set up blob detector
def blob_detector():
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;
        
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    # params.maxArea = 1000
        
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # params.maxCircularity = 1
        
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    # params.maxConvexity = 1
        
    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5
    # params.maxInertiaRatio = 1
    
    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

# Object detection
def object_detection(detect_obj, frame_name):
    # Keypoints
    kp = detect_obj.detect(frame_name)

    # Create frame and draw obatined keypoints
    img_kp = cv2.drawKeypoints(frame_name, kp, np.array([]), (0,0,255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Keypoints', img_kp)

    # If any blob(s) are found, sort them and return the biggest blob
    if len(kp):
        kp.sort(key = (lambda s : s.size))
        kp = kp[-1]     # Last element (biggest size)

        blob = True
        x = kp.pt[0]    # Blob center x-coordinate
        y = kp.pt[1]    # Blob center y-coordinate
        d = kp.size     # Blob diameter
        
    else:
        blob = False
        x = 0
        y = 0
        d = 0

    return (blob, x, y, d)

# Object detection    
def object_detection_init(detect_obj, frame_name, init):
    # Keypoints
    kp = detect_obj.detect(frame_name)

    # Create frame and draw obatined keypoints
    img_kp = cv2.drawKeypoints(frame_name, kp, np.array([]), (0,0,255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Keypoints', img_kp)

    # If any blob(s) are found, sort them and return the biggest blob
    if len(kp):
        kp.sort(key = (lambda s : s.size))
        kp = kp[-1]     # Last element (biggest size)

        blob = True
        x = kp.pt[0]    # Blob center x-coordinate
        y = kp.pt[1]    # Blob center y-coordinate
        d = kp.size     # Blob diameter
        
    else:
        blob = False
        x = 0
        y = 0
        d = 0
    while init:
        if cv2.waitKey(1) == ord('r'):
            init = False
            return (blob, x, y, d)
    return (blob, x, y, d)


# Object tracking
def init_func(detect_obj, frame_name, init):
    while init:
        (blob, x, y, d) = object_detection(detect_obj, frame_name)
        if cv2.waitKey(1) == ord('r'):
            init = False
            return (blob, x, y, d)
    

# Live camera feed
def cam_feed():
    # Create an object of web camera
    vid = cv2.VideoCapture(0)

    # Throw an error if video is not opened
    if not vid.isOpened():
        print('Could not open video')
        sys.exit()

    # Choose tracker
    tracker = tracker_type('KCF')
    # Tracker Parameters 
    detector = blob_detector()


    # Found blob
    blob = False
    init = True

    while (vid.isOpened()):
        
        # Read video frame
        ret, frame = vid.read()
        
        # Convert input frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Greyscale Frame', gray)

        # # Object detection
        # if not blob:
        #     (blob, x, y, d) = object_detection(detector, gray)

        (blob, x, y, d) = init_func(detector, gray, True)
        
        # Bounding box (x0, y0, w, h)
        bbox = (x - d/2, y - d/2, d, d)

        # Draw bounding box
        if ret:
            # Tracking success
            circ_x = int(x)
            circ_y = int(y)
            circ_r = int(d/2)
            cv2.circle(frame, (circ_x, circ_y), circ_r, (255, 0 , 0), 1, 1)
            
            # object_tracking(tracker, frame, bbox)
        else:
            cv2.putText(frame, 'Tracking Failure', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('Tracking Frame', frame)


        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(object_detection(detector, gray))
            print(circ_x, circ_y, circ_r)
            break
    

        


if __name__ == '__main__':
    # tracker('BOOSTING')
    # dummy()
    cam_feed()