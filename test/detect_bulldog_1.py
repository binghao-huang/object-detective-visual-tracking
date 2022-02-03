#!/usr/bin/env python
#!coding=utf-8
import rospy
import numpy as np
import PIL.Image as pilimage
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from yolo import YOLO

yolo = YOLO()

class image_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        global location_pub
        location_pub = rospy.Publisher("cv_bridge_location", Float64, queue_size=1)
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
        # print (cv_image.shape)
        cv_image = pilimage.fromarray(np.uint8(cv_image))
        cv_image, bbox_list, label_list = yolo.detect_image(cv_image)
        cv_image = np.array(cv_image)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        if type(label_list) != int: # 没检测到物体的时候，bbox_list和label_list为1
            num_of_obj = len(label_list)
            print('num_of_object:', num_of_obj)
            for i in range(num_of_obj):
                if 'person' in label_list[i]:
                    #frame = frame[bbox_list[i][0]: bbox_list[i][2], bbox_list[i][1]: bbox_list[i][3]]
                    object_center = (bbox_list[i][1]+bbox_list[i][3])*0.5
                    delta_x = object_center-320
                    print(delta_x)
                    location_pub.publish(delta_x)
                if 'bed' in label_list[i]:
                    print("yyy")
                pass
        
        else:
        	print('yolo未识别到任何物体')
        pass
        
        #for object in vision_database_dict:
        # 再将opencv格式额数据转换成ros image格式的数据发布
        # try:
        #     #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #     location_pub.publish(location_pub)
        # except CvBridgeError as e:
        #     print('e')

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
