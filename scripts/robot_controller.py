#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import numpy as np

from geometry_msgs.msg import Twist

class RobotController():
    def __init__(self,
                 kp_linear=1.0,
                 kp_angular=1.0,
                 dist_threshold=0.05,
                 angle_threshold=3*np.pi/180,
                 max_linear_speed=0.1,
                 max_angular_speed=0.5):
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.dist_threshold = dist_threshold
        self.angle_threshold = angle_threshold
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

        ## Transform Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        ## Publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def move_robot_forward(self, distance):
        '''Move the robot forward for the given distance.'''
        velocity = 0.1
        time = distance / velocity
        curr_time = rospy.Time.now().to_sec()
        vel_msg = Twist()
        rate = rospy.Rate(100)
        while rospy.Time.now().to_sec() - curr_time < time and not rospy.is_shutdown():
            vel_msg.linear.x = velocity
            self.cmd_vel_pub.publish(vel_msg)
            rate.sleep()
        vel_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(vel_msg)
    
    def rotate_robot(self, angle):
        '''Rotate the robot by the given angle. 
        Positive angle is counter-clockwise.'''
        angular_speed = 0.5
        time = angle / angular_speed
        curr_time = rospy.Time.now().to_sec()
        vel_msg = Twist()
        rate = rospy.Rate(100)
        while rospy.Time.now().to_sec() - curr_time < time and not rospy.is_shutdown():
            vel_msg.angular.z = angular_speed
            self.cmd_vel_pub.publish(vel_msg)
            rate.sleep()
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)

    @staticmethod
    def normalize_angle(angle):
        '''Helper function to Normalize the angle between -pi to pi.'''
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def move_to_waypoints(self, waypoints : list):
        '''Move the robot to the given waypoints.
        waypoints : List of tuples (x, y, theta)'''
        for pose in waypoints:
            self.move_to_pose(pose[0], pose[1], pose[2])

    def move_to_pose(self, x, y, theta):

        vel_msg = Twist()

        while not rospy.is_shutdown():
            ## Get the current pose.
            try:
                current_pose = self.tf_buffer.lookup_transform('map', 'robot_base', rospy.Time(0), rospy.Duration(5))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Failed to get the current pose.")
                continue

            current_trans = current_pose.transform.translation
            current_rot = current_pose.transform.rotation

            _, _, yaw = tf.transformations.euler_from_quaternion([current_rot.x, current_rot.y, current_rot.z, current_rot.w])

            x_current = current_trans.x
            y_current = current_trans.y
            theta_current = self.normalize_angle(yaw)

            ## Compute the position errors
            dx = x - x_current
            dy = y - y_current
            distance_to_goal = np.sqrt(dx**2 + dy**2)

            ## Compute the angle error
            angle_to_goal = np.arctan2(dy, dx)
            angle_error = np.arctan2(np.sin(angle_to_goal - theta_current), np.cos(angle_to_goal - theta_current))

            rospy.loginfo(f"Current: {x_current:.3f}, {y_current:.3f}, {theta:.3f} | Goal: {x:.3f}, {y:.3f}, {theta:.3f} | Dist: {distance_to_goal:.3f}, Angle: {angle_error:.3f}")

            if distance_to_goal > self.dist_threshold:
                ## Scale the linear speed based on the angle error.
                linear_speed_scaling_factor = min(1, np.exp(-5*abs(angle_error)))
                vel_msg.linear.x = np.clip(self.kp_linear * linear_speed_scaling_factor, 0, self.max_linear_speed)
                vel_msg.angular.z = np.clip(self.kp_angular * angle_error, -self.max_angular_speed, self.max_angular_speed)

            else:
                ## Close to the goal.
                vel_msg.linear.x = 0.0

                ## Calculate the angle error using arctan2 to avoid discontinuity at -pi and pi.
                angle_error = np.arctan2(np.sin(theta - theta_current), np.cos(theta - theta_current))

                ## Align with the goal orientation.
                if abs(angle_error) > self.angle_threshold:
                    vel_msg.angular.z = np.clip(self.kp_angular * angle_error, -self.max_angular_speed, self.max_angular_speed)
                else:
                    self.stop_robot()
                    rospy.loginfo("Goal Reached. Stopping the robot.")
                    break
            
            self.cmd_vel_pub.publish(vel_msg)

    def stop_robot(self):
        '''Stop the robot.'''
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(vel_msg)

def move_robot_in_a_square(controller):
    try:
        controller.move_robot_forward(1.5)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(1.5)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(1.5)
        controller.rotate_robot(np.pi/2)
        controller.move_robot_forward(1.5)
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")


def main():
    rospy.init_node('robot_controller')

    controller = RobotController()

    ## Ensure the robot stops if the node is killed.
    rospy.on_shutdown(controller.stop_robot)

    # ## Move Robot in a Square
    # move_robot_in_a_square(controller)

    ## Move to Waypoints
    waypoints = [
        (1.38, 0.46, 0.0),
        (1.38, 1.38, np.pi/2),
        (0.60, 1.18, np.pi),
    ]

    controller.move_to_waypoints(waypoints)

    rospy.signal_shutdown("Task Completed. Shutting down the node.")

if __name__ == '__main__':
    main()