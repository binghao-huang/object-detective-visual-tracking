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


# 回调函数，获取机器人 motor1 的joint sate
def RV2_motorjointstate_callback(data):
	global RV2_motor1_joint
	RV2_motor1_joint = data.position[0]


def active_cb(extra):
    rospy.loginfo("Goal pose being processed")


def feedback_cb(feedback):
    rospy.loginfo("Current location: "+str(feedback))


def done_cb(status, result):
    if status == 3:
        rospy.loginfo("Goal reached")
    if status == 2 or status == 8:
        rospy.loginfo("Goal cancelled")
    if status == 4:
        rospy.loginfo("Goal aborted")


# 回调函数，cvbridge获取中间相机的RGB图像
def ReceiveVideo_mid(data):
    global cv_image, bridge
    cv_image = bridge.compressed_imgmsg_to_cv2(data, 'bgr8')


# yolo目标识别程序，并返回当前识别结果，是否有人以及人的距离
def object_recognize():
    global delta_person, cv_image, label_list, person_count_total
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
        
        print (label_list, t_start, 'person_stay_count:', person_stay_count, 'person_count_total:', person_count_total) # for debug
    os._exit()

def motor1_move():
    time.sleep(1)
    global command_vel_pub_m, delta_person, RV2_motor1_joint, label_list
    delta_person = 0
    now = rospy.Time.now()
    motor_vel = JointState()
    motor_vel.header = Header()
    motor_vel.header.stamp = now
    motor_vel.header.frame_id = "motor1_link"
    motor_vel.name = ["motor1"]

    while not rospy.is_shutdown():
        # print('delta_person:', delta_person) # for debug
        #中间位置判断
        if -1.5 < RV2_motor1_joint < 1.5:
            #左转判断条件
            if delta_person > 200:
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            elif 80 < delta_person < 200:
                motor_vel.velocity = [(delta_person - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            #右转判断条件
            elif delta_person < -200: 
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            elif -200 < delta_person < -80: 
                motor_vel.velocity = [(delta_person + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            #停止判断条件
            elif -80 < delta_person < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)

        #左限位判断条件
        if 1.5 < RV2_motor1_joint:
            #左转判断条件
            if delta_person > 80:
                motor_vel.velocity = [0]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)
            #右转判断条件
            elif delta_person < -200: 
                motor_vel.velocity = [-0.48]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            elif -200 < delta_person < -80: 
                motor_vel.velocity = [(delta_person + 40) * 0.003]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            #停止判断条件
            elif -80 < delta_person < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)

        #右限位判断条件
        if RV2_motor1_joint < -1.5:
            #左转判断条件
            if delta_person > 200:
                motor_vel.velocity = [0.48]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            elif 80 < delta_person < 200:
                motor_vel.velocity = [(delta_person - 40) * 0.003]
                print (motor_vel)
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            #右转判断条件
            elif delta_person < -80: 
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(2)

            #停止判断条件
            elif -80 < delta_person < 80:
                motor_vel.velocity = [0]
                command_vel_pub_m.publish(motor_vel)
                time.sleep(0.5)
             
        else:
            motor_vel.velocity = [0]
            command_vel_pub_m.publish(motor_vel)
            time.sleep(0.5)


# 底盘运动程序
def base_move():
    global person_count_total, trans, rot, tf_listener
    flag_on_mainroad = True
    person_stay_threshold = 2 # 参数阈值,判断周期内识别到人的次数的界定值

    try:
        (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.loginfo("tf Error")

    while not rospy.is_shutdown():
        # 如果人在镜头中出现一定时间,并且机器人当前在主路中央,执行靠边运动
        if person_count_total > person_stay_threshold and flag_on_mainroad:
            try:
                (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("tf Error")
            print('flag', flag_on_mainroad)
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = trans[0] + 0.2
            goal.target_pose.pose.position.y = trans[1]
            goal.target_pose.pose.position.z = 0
            goal.target_pose.pose.orientation.x = 0
            goal.target_pose.pose.orientation.y = 0
            goal.target_pose.pose.orientation.z = rot[2]
            goal.target_pose.pose.orientation.w = rot[3]
            
            flag_on_mainroad = False
            print('flag', flag_on_mainroad)
            print(goal)
            
            navclient.send_goal(goal,done_cb)
            finished = navclient.wait_for_result()

            # flag_on_mainroad = False

        # 如果人不在镜头中一段时间,并且机器人不在路中央(已执行靠边),回归主路
        elif person_count_total < person_stay_threshold and not(flag_on_mainroad):
            try:
                (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("tf Error")

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = trans[0] + 0.5
            goal.target_pose.pose.position.y = trans[1] + 0.5
            goal.target_pose.pose.position.z = 0.0
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.0
            goal.target_pose.pose.orientation.w = 1.0

            navclient.send_goal(goal)

            flag_on_mainroad = True

        # 如果人在镜头中出现一段时间,并且机器人不在路中央(已执行靠边),pass等待
        elif person_count_total > person_stay_threshold and not(flag_on_mainroad):
            pass
                        
        time.sleep(0.2)


def Goal():
    global label_list, trans, rot, tf_listener
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = 0.3
    goal.target_pose.pose.position.y = 0
    goal.target_pose.pose.position.z = 0.0
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.0
    goal.target_pose.pose.orientation.w = 1.0

    navclient.send_goal(goal,done_cb,active_cb, feedback_cb)
    finished = navclient.wait_for_result()

    if not finished:
        rospy.logerr("Action server not available!")
    else:
        rospy.loginfo ( navclient.get_result())



if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("vision_based_move")
        rospy.loginfo("Starting vision based move node")
        global command_vel_pub_m, delta_person, trans, rot, tf_listener, navclient, bridge, RV2_motor1_joint
        
        # 发布电机速度
        command_vel_pub_m = rospy.Publisher('/motor_control/input/velocity', JointState, queue_size = 100, latch=True)
        # 订阅躯干点电机位置信息
        rospy.Subscriber('/joint_states_motor', JointState, RV2_motorjointstate_callback)
        # 订阅相机图像
        rospy.Subscriber('/mid_camera/color/image_raw/compressed', CompressedImage, ReceiveVideo_mid)
        # 实例化tf订阅, cv bridge
        tf_listener = tf.TransformListener()
        bridge = CvBridge()
        # 底盘导航 action启动
        navclient = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        navclient.wait_for_server()

        # 定义yolo识别子程序
        t_object_recognize = threading.Thread(target = object_recognize)
        t_object_recognize.start()
        
        # # 定义躯干运动子进程
        t_motor1 = threading.Thread(target = motor1_move)
        t_motor1.start()

        # 定义底盘运动子进程
        t_base = threading.Thread(target = base_move)
        t_base.start()

        # t_goal = threading.Thread(target = Goal)
        # t_goal.start()

        # t_goal = threading.Thread(target = Goal_test)
        # t_goal.start()

    except KeyboardInterrupt:
        print("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()
        sys.exit()
