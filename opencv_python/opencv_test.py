import cv2
import sys

if __name__ == '__main__' :

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

    vid = cv2.VideoCapture(0)
    bbox = 0;
    # while True:
    #     ret, frame = cap.read()

    # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', frame)
    # # cv2.imshow('gray frame',gray)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cap.release()

    # cv2.destroyAllWindows()

    while (vid.isOpened()):
        
        if not bbox:
            ok, frame = vid.read()
            bbox = cv2.selectROI(frame, False)
            ok = tracker.init(frame, bbox)
            if not ok:
                break
        
        ok, frame = vid.read()
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 1, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2) 
        

        # Display result
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    vid.release()

    cv2.destroyAllWindows() 



