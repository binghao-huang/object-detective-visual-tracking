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
bridge = CvBridge()
def send():
    rospy.Subscriber('/mid_camera/color/image_raw/compressed', CompressedImage, ReceiveVideo_right)
    rospy.spin()

def ReceiveVideo_right(data):
    global cv_image
    # print(1)
    cv_image = bridge.compressed_imgmsg_to_cv2(data, 'bgr8')

def main():
    global delta_x,cv_image
    time.sleep(4)
    fps = 0
    while not rospy.is_shutdown():
        t1 = time.time()
        # 读取某一帧
        frame = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = pilimage.fromarray(np.uint8(frame))
        # 进行检测
        frame, bbox_list, label_list = yolo.detect_image(frame)
        frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(frame.shape)

        cv2.imshow("video",frame)
        cv2.waitKey(3)
        # c= cv2.waitKey(1) & 0xff 
        # if c==27:
        #     break
    
        if type(label_list) != int: # 没检测到物体的时候，bbox_list和label_list为1
            num_of_obj = len(label_list)
            #print('num_of_object:', num_of_obj)
            #确定跟踪物体与图像中点的相对坐标
            for i in range(num_of_obj):
                if 'banana' in label_list[i]:
                    object_center = (bbox_list[i][1]+bbox_list[i][3])*0.5
                    delta_x = 320-object_center
                    #print(delta_x)
                    #return delta_x
                    # location_pub.publish(delta_x)
                    #motor1_move()
                elif 'bed' in label_list[i]:
                    print("yyy")
                pass
        else:
            print('yolo未识别到任何物体')
        pass

def motor1_move():
    time.sleep(1)
    global command_vel_pub_m, delta_x, RV2_motor1_joint
    delta_x = 0
    now = rospy.Time.now()
    motor_vel = JointState()
    motor_vel.header = Header()
    motor_vel.header.stamp = now
    motor_vel.header.frame_id = "bulldog"
    motor_vel.name = ["motor1"]
    # rospy.Subscriber('/joint_states_motor',JointState,RV2_motorjointstate_callback)
    while not rospy.is_shutdown():
        print(delta_x)
        #中间位判断
        if -1.5 < RV2_motor1_joint < 1.5:
                #左转判断条件
            if delta_x > 200:
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif 80 < delta_x < 200:
                motor_vel.velocity = [(delta_x - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -200: 
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif -200 < delta_x < -80: 
                motor_vel.velocity = [(delta_x + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
        #左限位判断条件
        if 1.5 < RV2_motor1_joint:
                #左转判断条件
            if delta_x > 80:
                motor_vel.velocity = [0]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -200: 
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif -200 < delta_x < -80: 
                motor_vel.velocity = [(delta_x + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)
        #右限位判断条件
        if RV2_motor1_joint < -1.5:
                #左转判断条件
            if delta_x > 200:
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            elif 80 < delta_x < 200:
                motor_vel.velocity = [(delta_x - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_x < -80: 
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #停止判断条件
            elif -80 < delta_x < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)                
        else:
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
    # 初始化ros节点
    rospy.init_node("cv_bridge_test")
    rospy.loginfo("Starting cv_bridge_test node")
    global command_vel_pub_m, delta_x
    #创建发布者
    command_vel_pub_m = rospy.Publisher('/motor_control/input/velocity', JointState, queue_size = 100, latch=True)
    #订阅躯干点击位置信息
    rospy.Subscriber('/joint_states_motor',JointState,RV2_motorjointstate_callback)

    #定义yolo识别子程序
    t_send = threading.Thread(target = send)
    t_send.start()
    t_main = threading.Thread(target=main)
    t_main.start()
    
    #time.sleep(2)
    # 定义躯干运动子进程
    t_motor1 = threading.Thread(target = motor1_move)
    t_motor1.start()

    rospy.spin()

    # except KeyboardInterrupt:
    #     print("Shutting down cv_bridge_test node.")
    #     cv2.destroyAllWindows()
