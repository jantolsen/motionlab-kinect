import cv2
import sys
import numpy as np

detect_init = True

# Set up tracker
def tracker_type(name):
    if name == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
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

def detector_func(video_object):
    global detect_init
    # Detector Parameters 
    detector = blob_detector()

    while video_object.isOpened():
        
        # Read video frame
        ret, frame = video_object.read()

        # Convert input frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Show frames
        # cv2.imshow('Original Frame', frame)
        # cv2.imshow('Grayscale Frame', gray)
        # cv2.imshow('HSV Frame', hsv)

        # Keypoints
        kp = detector.detect(gray)

        # Create frame and draw obatined keypoints
        img_kp = cv2.drawKeypoints(frame, kp, np.array([]), (0,0,255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        # Show frame with keypoint(s)        
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
        
        # Break loop if not a inital detection and a blob is found and 
        if not detect_init and blob:
            return (x, y, d)

        # Break loop on keystroke ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detect_init = False
            return (x, y, d)

# Object tracking       
def tracker_func(video_object):
    
    reset = False
    # Read video frame
    ret, frame = video_object.read()

    # Choose tracker
    tracker = tracker_type('KCF')

    # Initial detection
    # detect_init = True
    (x, y, d) = detector_func(video_object)
    
    # Bounding box (x0, y0, w, h)
    bbox = (x - d/2, y - d/2, d, d)
    ret = tracker.init(frame, bbox)

    while (video_object.isOpened()):
        # Read a new frame
        ok, frame = video_object.read()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            c_x = int(bbox[0] + d/2)    # Circle center x-coordinate
            c_y = int(bbox[1] + d/2)    # Circle center y-coordinate
            c_r = int(bbox[3]/2)        # Circle radius
            cv2.circle(frame, (c_x, c_y), c_r, (255, 0 , 0), 1, 1)
        else :
            # Tracking failure
            cv2.putText(frame, 'Tracking Failure', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            # detect_init = False
            # tracker_func(video_object)
            # Reset Detect object
            reset = True
            # (x, y, d) = detector_func(video_object)
            # print('I am trying')
            # bbox = (x - d/2, y - d/2, d, d)
            # ret = tracker.init(frame, bbox)

        cv2.imshow('Tracking Frame', frame)

        if reset:
            video_object.release()
            # cam_feed()

            # tracker_func(video_object)
        # Break loop on keystroke ('r')
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break


# Live camera feed
def cam_feed():
    # Create an object of web camera
    video = cv2.VideoCapture(0)

    # Throw an error if video is not opened
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # detect_init = True
    # (x, y, d) = detector_func(video, detect_init)
    # print(x,y,d)
    # print(detect_init)
    tracker_func(video)
    # tracker = tracker_type('KCF')
    # print(tracker)
if __name__ == '__main__':
    # detect_init = True
    cam_feed()
    cv2.destroyAllWindows()
