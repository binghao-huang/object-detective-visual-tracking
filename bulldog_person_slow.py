#!/usr/bin/env python3
#!coding=utf-8
import rospy
import numpy as np
import PIL.Image as pilimage
import actionlib
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from yolo import YOLO
from tf_conversions import transformations
import tf
import os, sys

from sensor_msgs.msg import Joy
from std_msgs.msg import String 
from geometry_msgs.msg import Twist
from tf.transformations import *
from math import pi
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped,Twist
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from threading import Thread
import threading


# 回调函数，cvbridge获取中间相机的RGB图像
def ReceiveVideo_mid(data):
    global cv_image, bridge
    cv_image = bridge.compressed_imgmsg_to_cv2(data, 'bgr8')


# yolo目标识别程序，并返回当前识别结果，是否有人以及人的距离
def object_recognize():
    global delta_person, cv_image, label_list, person_count_total, bridge
    fps = 0
    count_period = 3 # 目标检测计数周期
    person_stay_count = 0 
    person_count_total = 0
    t_start = time.time()
    yolo = YOLO()
    time.sleep(1)
    while not rospy.is_shutdown():
        # 读取某一帧
        frame = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = pilimage.fromarray(np.uint8(frame))
        # 进行检测
        frame, bbox_list, label_list = yolo.detect_image(frame)
        frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        # fps  = ( fps + (1./(time.time()-t_start)) ) / 2
        # print("fps= %.2f"%(fps))
        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print(frame.shape)

        cv2.imshow("video",frame)
        cv2.waitKey(3)
        
        if type(label_list) != int: # 没检测到物体的时候，bbox_list和label_list为1
            num_of_obj = len(label_list)
            #确定跟踪物体与图像中点的相对坐标
            for i in range(num_of_obj):
                if 'person' in label_list[i]:
                    person_stay_count = person_stay_count + 1
                    object_center = (bbox_list[i][1]+bbox_list[i][3])*0.5
                    delta_person = 320-object_center
        else:
            print('yolo未识别到任何物体')

        # 每一 period 重新计时，识别目标计数person_stay_count归零, 同时将前一计时周期内识别到人的置信计数存储到person_stay_count
        if time.time() - t_start > count_period:
            t_start = time.time()
            person_count_total = person_stay_count
            person_stay_count = 0
        else:
            pass
        
        print (label_list) # for debug
    os._exit()

# 底盘运动程序
def base_move():
    global person_count_total, label_list, person_count_total
    person_stay_threshold = 2 # 参数阈值,判断周期内识别到人的次数的界定值
    vel_msg = Twist()
    while not rospy.is_shutdown():
        # 如果人在镜头中出现一定时间,低速
        if person_count_total > person_stay_threshold:
            vel_msg.linear.x = 0.05
			vel_msg.angular.z = 0
			bulldog_vel_pub.publish(vel_msg)
			rospy.loginfo("Publsh bulldog velocity command[%0.2f m/s, %0.2f rad/s]",
                                vel_msg.linear.x, vel_msg.angular.z)

            # flag_on_mainroad = False

        # 如果人不在镜头中一段时间,正常速度
        elif person_count_total < person_stay_threshold:
            vel_msg.linear.x = 0.1
			vel_msg.angular.z = 0
			bulldog_vel_pub.publish(vel_msg)
			rospy.loginfo("Publsh bulldog velocity command[%0.2f m/s, %0.2f rad/s]",
                                vel_msg.linear.x, vel_msg.angular.z)
        # 如果人在镜头中出现一段时间,并且机器人不在路中央(已执行靠边),pass等待
                        
        time.sleep(0.2)


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("vision_based_move")
        rospy.loginfo("Starting vision based move node")
        global command_vel_pub, delta_person, bridge，person_count_total, label_list, person_count_total
        bulldog_vel_pub = rospy.Publisher('/bulldog_velocity_controller/cmd_vel', Twist, queue_size=10)
        # 订阅相机图像
        rospy.Subscriber('/mid_camera/color/image_raw/compressed', CompressedImage, ReceiveVideo_mid)
        # 实例化tf订阅, cv bridge
        bridge = CvBridge()

        # 定义yolo识别子程序
        t_object_recognize = threading.Thread(target = object_recognize)
        t_object_recognize.start()

        # 定义底盘运动子进程
        t_base = threading.Thread(target = base_move)
        t_base.start()


    except KeyboardInterrupt:
        print("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()
        sys.exit()
