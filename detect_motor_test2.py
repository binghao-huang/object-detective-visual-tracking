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

from sensor_msgs.msg import Joy
from std_msgs.msg import String 
from geometry_msgs.msg import Twist
from tf.transformations import *
from math import pi
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from threading import Thread
import threading

global RV2_motor1_joint

yolo = YOLO()

class image_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        global delta_x
        # location_pub = rospy.Publisher("cv_bridge_location", Float64, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/mid_camera/color/image_raw/compressed", CompressedImage, self.callback)
    def callback(self,data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        global delta_x
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print('e')
        #BGR转RGB格式
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        #cv格式转image
        cv_image = pilimage.fromarray(np.uint8(cv_image))
        #进行yolo语音识别，提取框位置信息与识别物体信息
        cv_image, bbox_list, label_list = yolo.detect_image(cv_image)
        #image转cv格式
        cv_image = np.array(cv_image)
        #RGB在转BGR格式
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_RGB2BGR)
        #显示识别后cv图像
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        if type(label_list) != int: # 没检测到物体的时候，bbox_list和label_list为1
            num_of_obj = len(label_list)
            #print('num_of_object:', num_of_obj)
            #确定跟踪物体与图像中点的相对坐标
            for i in range(num_of_obj):
                if 'banana' in label_list[i]:
                    object_center = (bbox_list[i][1]+bbox_list[i][3])*0.5
                    delta_x = 320-object_center
                    # print(delta_x)
                    return delta_x
                    # location_pub.publish(delta_x)
                    #motor1_move()
                elif 'bed' in label_list[i]:
                    print("yyy")
                pass
        else:
        	print('yolo未识别到任何物体')
        pass

def judge_bed():
    global delta_x
    image_converter()

def motor1_move():
    time.sleep(1)
    global command_vel_pub_m, delta_x, RV2_motor1_joint
    # rospy.Subscriber('/joint_states_motor',JointState,RV2_motorjointstate_callback)
    while not rospy.is_shutdown():
        print(delta_x)
        #中间位判断
        if -1.5 < RV2_motor1_joint < 1.5:
                #左转判断条件
            if delta_x > 200:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif 80 < delta_x < 200:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [(delta_x - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -200: 
                print ("b")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif -200 < delta_x < -80: 
                print ("b")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [(delta_x + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                time.sleep(2)
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
        #左限位判断条件
        if 1.5 < RV2_motor1_joint:
                #左转判断条件
            if delta_x > 80:
                print("a")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -200: 
                print ("b")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif -200 < delta_x < -80: 
                print ("b")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [(delta_x + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)
        #右限位判断条件
        if RV2_motor1_joint < -1.5:
                #左转判断条件
            if delta_x > 200:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif 80 < delta_x < 200:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [(delta_x - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -80: 
                print ("b")
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)                
        else:
                now = rospy.Time.now()
                motor_vel = JointState()
                motor_vel.header = Header()
                motor_vel.header.stamp = now
                motor_vel.header.frame_id = "bulldog"
                motor_vel.name = ["motor1"]
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)

       
        #for object in vision_database_dict:
        # 再将opencv格式额数据转换成ros image格式的数据发布
        # try:
        #     #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #     location_pub.publish(location_pub)
        # except CvBridgeError as e:
        #     print('e')
def RV2_motorjointstate_callback(data):
    # 定义RV2 motor数据全局变量，进行赋值
	global RV2_motor1_joint
	RV2_motor1_joint = data.position[0]
	print(RV2_motor1_joint)

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("cv_bridge_test")
        rospy.loginfo("Starting cv_bridge_test node")
        global command_vel_pub_m, delta_x
        #创建发布者
        command_vel_pub_m = rospy.Publisher('/motor_control/input/velocity', JointState, queue_size = 100, latch=True)
        #订阅躯干点击位置信息
        rospy.Subscriber('/joint_states_motor',JointState,RV2_motorjointstate_callback)

        #定义yolo识别子程序
        t_judge_bed = threading.Thread(target = judge_bed)
        t_judge_bed.start()
        
        time.sleep(2)
        # 定义躯干运动子进程
        t_motor1 = threading.Thread(target = motor1_move)
        t_motor1.start()

        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()
