import numpy as numpy
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2

class KinectV2(object):
    
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                PyKinectV2.FrameSourceTypes_BodyIndex |
                                                PyKinectV2.FrameSourceTypes_Depth |
                                                PyKinectV2.FrameSourceTypes_Infrared)
        self.rgb_Fx = 1081.37
        self.rgb_Fy = 1081.37
        self.rgb_Cx = 959.5
        self.rgb_Cy = 539.5

        

        self.depth_Fx = 366.193
        self.depth_Fy = 366.193
        self.depth_Cx = 256.684
        self.depth_Cy = 207.085
        self.depth_k1 = 0.0905474
        self.depth_k2 = -0.26819
        self.depth_k3 = 0.0950862
        self.depth_p1 = 0.0
        self.depth_p2 = 0.0

        self.ir_img_size = (424,512)
        self.rgb_img_size = (1080,1920)
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
        depthFrame = depthFrame.reshape(self.ir_img_size)
        bodyIndexFrame = bodyIndexFrame.reshape(self.ir_img_size)
        infraredFrame = infraredFrame.reshape(self.ir_img_size)

        return (colorFrame, depthFrame, bodyIndexFrame, infraredFrame)

def main():
    # Drawing colors
    green_color = (0,255,0)    # BGR
    red_color = (0, 0, 255)    # BGR

    kinect = KinectV2()

    while True:
        frame = kinect.take_pic()

        color_frame = frame[0]
        depth_frame = frame[1]
        body_frame = frame[2]
        ir_frame = frame[3]

        cv2.imshow('color', color_frame)
        # cv2.imshow('depth', depth_frame)
        cv2.imshow('IR', ir_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            kinect.close()
            break

if __name__ == '__main__':
    main()