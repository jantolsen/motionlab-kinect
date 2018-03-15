import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import time

ir_img_size = (424,512)
rgb_img_size = (1080,1920)
number_of_calib_frame = 100

def ir_frame_to_jpg(IRFrame):
    IRFrame = IRFrame.reshape(ir_img_size)
    IRFrame = np.uint8(IRFrame/256)

    jpgIRFrame = np.zeros((424, 512, 3), np.uint8)
    jpgIRFrame[:,:,0] = IRFrame
    jpgIRFrame[:,:,1] = IRFrame
    jpgIRFrame[:,:,2] = IRFrame

    return jpgIRFrame


kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Infrared |
                                         PyKinectV2.FrameSourceTypes_Depth)

i = 0
mytime = time.time()

redAlert = np.zeros((424, 512, 3), np.uint8)
redAlert[:,:,2] = 255

while(True):
    mytime = time.time()

    while(cv2.waitKey(27) != 27):
        cv2.imshow('IR', ir_frame_to_jpg(kinect.get_last_infrared_frame()))

    cv2.imshow('IR', redAlert)
    cv2.waitKey(1)

    IRFrame = kinect.get_last_infrared_frame()
    jpgIRFrame = ir_frame_to_jpg(IRFrame)
    irFilePath = 'C:/Users/Workstation/Documents/Kinect Calibration/IR_Frame/' + str(i) + '.jpg'
    cv2.imwrite(irFilePath, jpgIRFrame)

    colorFrame = kinect.get_last_color_frame()
    colorFrame = colorFrame.reshape(1080, 1920, 4)
    rgbFilePath = 'C:/Users/Workstation/Documents/Kinect Calibration/RGB_Frame/' + str(i) + '.jpg'
    cv2.imwrite(rgbFilePath, colorFrame)

    for j in range(0,number_of_calib_frame):
        time.sleep(0.03)
        depthFilePath = 'C:/Users/Workstation/Documents/Kinect Calibration/DEPTH_Frame/' + str(i) + '_' + str(j) + '.npy'
        depthFrame = kinect.get_last_depth_frame()
        depthFrame = depthFrame.reshape(ir_img_size)
        np.save(depthFilePath, depthFrame)

    i = i + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        kinect.close()
        break
print('done')
