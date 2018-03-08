import cv2
import numpy as np

vid = cv2.VideoCapture(0)

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
    
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;
    
# Filter by Area.
params.filterByArea = True
params.minArea = 500
    
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
    
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
    
# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)
color_red = (0,0,255)
blob = False
while (vid.isOpened()):
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray frame',gray)


    kp = detector.detect(gray)
    
    im_kp = cv2.drawKeypoints(gray, kp, np.array([]), color_red, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('kp', im_kp)
    
    
    if len(kp):
    #         # if more than four blobs, keep the four largest
        kp.sort(key=(lambda s: s.size))
        kp=kp[-1]   # Last element (biggest size)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(kp.size)
d = kp.size  # Diameter of blob
bbox = (kp.pt[0] - d/2, kp.pt[1] - d/2, d, d) # (x0, y0, w, h)

ok = tracker.init(frame, bbox)
while True:
    
    # Read a new frame
    ok, frame = vid.read()

    ok, bbox = tracker.update(frame)

        # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    # Display result
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

    #         print(kp.pt)
    #         # for p in kp:
    #         #     print(p.pt)
    # for p in kp:
    #     print(p.pt

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# for p in kp:
#     print(p.pt)
# print(kp(0))
# for kp in kp:
#         d = kp.size
# print(dir(kp))
# print(kp.size)
# print(kp[1].size)
vid.release()
cv2.destroyAllWindows()