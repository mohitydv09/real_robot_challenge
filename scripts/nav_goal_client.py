#! /usr/bin/env python3
# The publisher to specify a goal for the robot to go
import rospy
import math
import sys
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def movebase_client(goal_x=0.8, goal_y=-1.7, goal_theta=0):

	# Create a client to request a service from move_base node
	# to move the robot
	client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
	client.wait_for_server()

	# Create the goal 
	goal = MoveBaseGoal()
	now = rospy.get_rostime()
	goal.target_pose.header.stamp.secs = now.secs
	goal.target_pose.header.stamp.nsecs = now.nsecs
	goal.target_pose.header.frame_id = "map"

	goal.target_pose.pose.position.x = goal_x
	goal.target_pose.pose.position.y = goal_y
	goal.target_pose.pose.position.z = 0

	goal.target_pose.pose.orientation.x = 0.0
	goal.target_pose.pose.orientation.y = 0.0
	goal.target_pose.pose.orientation.z = math.sin(goal_theta/2)
	goal.target_pose.pose.orientation.w = math.cos(goal_theta/2)

	# Send the goal and wait for the result
	client.send_goal(goal)
	wait = client.wait_for_result()

	if not wait:
		rospy.logerr("Action server not available!")
		rospy.signal_shutdown("Action server available!")
	else:
		return client.get_result()



if __name__ == '__main__':
	try:
		rospy.init_node('nav_goal_publisher')
		goal_x = float(sys.argv[1])
		goal_y = float(sys.argv[2])
		goal_theta = float(sys.argv[3]) * (math.pi/180)
		result = movebase_client(goal_x, goal_y, goal_theta)
		if result:
			rospy.loginfo("Goal Navigation Execution Done!")
			rospy.loginfo(result)
	except rospy.ROSInterruptException:
		rospy.loginfo("Interruption Occurred in nav_goal_publisher")
