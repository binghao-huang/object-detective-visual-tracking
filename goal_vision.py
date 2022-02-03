#!/usr/bin/env python3
#-*- coding:utf-8   -*-
 
 
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf
 

def goal_pose():
	goal_pose=MoveBaseGoal()
	goal_pose.target_pose.header.frame_id="map"
	goal_pose.target_pose.pose.position.x=3
	goal_pose.target_pose.pose.position.y=0
	goal_pose.target_pose.pose.position.z=0
	goal_pose.target_pose.pose.orientation.x=0
	goal_pose.target_pose.pose.orientation.y=0
	goal_pose.target_pose.pose.orientation.z=0
	goal_pose.target_pose.pose.orientation.w=1
	print (goal_pose)
	return goal_pose
 
 
if __name__ == "__main__": 

	rospy.init_node('patrol')
 
    #创建MoveBaseAction client
	client=actionlib.SimpleActionClient('move_base',MoveBaseAction)
    #等待MoveBaseAction server启动
	client.wait_for_server()
	while not rospy.is_shutdown():
			goal=goal_pose()
			client.send_goal(goal)
			client.wait_for_result()
