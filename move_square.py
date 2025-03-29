#! /usr/bin/env python3

import os
import rospy
import numpy as np
import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, TransformStamped

class RobotController:
    def __init__(self):
        rospy.loginfo('Robot Controller Node Started.')
        self.robot_pose_raw_odom = np.array([[0.0],
                                             [0.0],
                                             [0.0]])

        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber for joint states
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        self.rate = rospy.Rate(20)

        self.last_time = None

        ## Transfrom broadcaster
        self.br = tf2_ros.TransformBroadcaster()

    def joint_state_callback(self, msg):
        """Callback function for joint state subscriber."""
        if self.last_time is None:
            self.last_time = msg.header.stamp.to_sec()
            return
        current_time = msg.header.stamp.to_sec()
        dt = current_time - self.last_time
        self.last_time = current_time

        w_left, w_right = msg.velocity[0], msg.velocity[1]

        ## Robot Model Constants, There values are taken from the robot model.
        wheel_radius = 0.033 ## 33 mm
        half_wheel_track = 0.144 ## 144 mm
        
        ## Convert the wheel velocities to linear and angular velocities.
        v = wheel_radius * (w_left + w_right) / 2
        w = wheel_radius * (w_right - w_left) / (2 * half_wheel_track)

        ## Move throught motion model.
        self.robot_pose_raw_odom[0, 0] += v * np.cos(self.robot_pose_raw_odom[2, 0]) * dt
        self.robot_pose_raw_odom[1, 0] += v * np.sin(self.robot_pose_raw_odom[2, 0]) * dt
        self.robot_pose_raw_odom[2, 0] += w * dt

        ## Publish the transform.
        self.publish_raw_odom_tf()

    def publish_raw_odom_tf(self):
        """Publish the odometry transform."""
        x, y, theta = self.robot_pose_raw_odom[0, 0], self.robot_pose_raw_odom[1, 0], self.robot_pose_raw_odom[2, 0]
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = "robot_base_raw_odom"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(theta / 2)
        t.transform.rotation.w = np.cos(theta / 2)
        self.br.sendTransform(t)
        
    def move_robot_forward(self, distance):
        """Move the robot forward for the given distance."""
        velocity = 0.1
        time = distance / velocity
        curr_time = rospy.Time.now().to_sec()
        vel_msg = Twist()
        while rospy.Time.now().to_sec() - curr_time < time and not rospy.is_shutdown():
            vel_msg.linear.x = velocity
            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()
        vel_msg.linear.x = 0
        self.cmd_vel_pub.publish(vel_msg)

    def rotate_robot(self, angle):
        '''Rotate the robot by the given angle. 
        Positive angle is counter-clockwise.'''
        angular_speed = 0.5
        time = angle / angular_speed
        curr_time = rospy.Time.now().to_sec()
        vel_msg = Twist()
        while rospy.Time.now().to_sec() - curr_time < time and not rospy.is_shutdown():
            vel_msg.angular.z = angular_speed
            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)
        self.rate.sleep()

    def stop_robot(self):
        """Stop the robot."""
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(vel_msg)

def move_robot_in_a_square(controller, side_length=1.0):
    try:
        controller.move_robot_forward(side_length)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(side_length)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(side_length)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(side_length)
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")

def main():
    rospy.init_node('robot_square_mover')

    controller = RobotController()

    rospy.on_shutdown(controller.stop_robot)

    move_robot_in_a_square(controller, side_length=1.0)

    rospy.spin()

if __name__=='__main__':
    main()