#!/usr/bin/env python
#!coding=utf-8
import rospy
import numpy as np
import PIL.Image as pilimage
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from yolo import YOLO

yolo = YOLO()

class image_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        #self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1)
        #self.image_pub = rospy.Publisher("cv_bridge_image", MultiArrayLayout, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/mid_camera/color/image_raw/compressed", CompressedImage, self.callback)
    def callback(self,data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print('e')
        
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        cv_image = pilimage.fromarray(np.uint8(cv_image))
        cv_image, bbox_list, label_list = yolo.detect_image(cv_image)
        cv_image = np.array(cv_image)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        # 再将opencv格式额数据转换成ros image格式的数据发布
        #try:
        #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            #self.image_pub.publish(self.bridge.cv2_to_imgmsg(bbox_list, MultiArrayLayout))
        #except CvBridgeError as e:
        #    print('e')

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("cv_bridge_test")
        rospy.loginfo("Starting cv_bridge_test node")
        image_converter()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()
