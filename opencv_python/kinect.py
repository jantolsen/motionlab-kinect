import numpy as numpy
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import os

class KinectV2(object):
    
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                 PyKinectV2.FrameSourceTypes_BodyIndex |
                                                 PyKinectV2.FrameSourceTypes_Depth |
                                                PyKinectV2.FrameSourceTypes_Infrared)

    def close(self):
        self.kinect.close()

    def take_pic(self):
        # get frames from kinect 
        colorFrame = self.kinect.get_last_color_frame()
        depthFrame = self.kinect.get_last_depth_frame()
        bodyIndexFrame = self.kinect.get_last_body_index_frame()
        infraredFrame = self.kinect.get_last_infrared_frame()

        # reshape to 2-D space
        colorFrame = colorFrame.reshape(1080,1920,4)
        depthFrame = depthFrame.reshape(424,512)
        bodyIndexFrame = bodyIndexFrame.reshape(424,512)
        infraredFrame = infraredFrame.reshape(424,512)

        return (colorFrame, depthFrame, bodyIndexFrame, infraredFrame)

kinect = KinectV2()
print('hello world')
print(kinect)

while True:
    frame = kinect.take_pic()
    color = cv2.resize( frame[0], (1280, 720))

    cv2.imshow('test', color)

    # cv2.imshow('color', frame[0])
    # cv2.imshow('depth', frame[1])
    # cv2.imshow('body', frame[2])
    cv2.imshow('infra', frame[3])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        kinect.close()
        break