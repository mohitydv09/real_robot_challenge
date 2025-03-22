#! /usr/bin/env python3
import rospy
import tf
import tf2_ros
import numpy as np
import open3d as o3d
import threading

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, LaserScan, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist

import laser_geometry.laser_geometry as lg


class KalmanFilter:
    def __init__(self):
        self.x = np.array([[0], 
                           [0], 
                           [0]])  ## State matrix.

        self.P = np.eye(3) * 0.1            ## Covariance Matrix
        self.Q = np.eye(3) * 0.01           ## Process Noise
        self.R = np.eye(3) * 1.0            ## Measurement Noise

    def predict(self, v, w, dt):
        theta = self.x[2, 0]
        ## Motion Model Matrix
        F = np.eye(3)
        B = dt * np.array([ [np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [0,             1]])
        u = np.array([[v],
                      [w]])

        ## Predict the state.
        self.x = F @ self.x + B @ u
        self.x[2, 0] = self.normalize_angle(self.x[2, 0])

        ## Predict the Covariance Matrix
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z):
        '''Update the state and covariance matrix based on z. Where z is measurment.'''

        ## Measurement Model Matrix
        H = np.eye(3)

        ## Compute the Kalman Gain.
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        ## Update the state.
        self.x = self.x + K @ (z - H @ self.x)
        self.x[2, 0] = self.normalize_angle(self.x[2, 0])

        ## Update the Covariance Matrix
        self.P = (np.eye(3) - K @ H) @ self.P

    def normalize_angle(self, angle):
        '''Normalize the angle between -pi to pi.'''
        return (angle + np.pi) % (2 * np.pi) - np.pi


class LocalizationNode():
    def __init__(self):
        # rospy.init_node('localization_node')
        rospy.loginfo('Localization Node Started.')

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=50)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=10)
        self.transformed_pointcloud_pub = rospy.Publisher('/transformed_pointcloud', PointCloud2, queue_size=10)

        self.kf = KalmanFilter()
        self.last_time = None

        ## Load the presaved Pointcloud map
        self.map_pcd = o3d.io.read_point_cloud('arena.pcd') 
        self.initialization_transform = None

        ## Helper for LaserScan to PointCloud Conversions.
        self.laser_projector = lg.LaserProjection()

        ## Transform Publisher.
        self.br = tf2_ros.TransformBroadcaster()

        ## Transform Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # rospy.spin()

    def odom_callback(self, msg : Odometry):
        '''Run the predict step of KF each time odometry data is recieved.'''
        ## If this is first run just update the time.
        if self.last_time is None:
            self.last_time = msg.header.stamp.to_sec()
            return
        
        current_time = msg.header.stamp.to_sec()
        dt = current_time - self.last_time
        self.last_time = current_time

        ## Get the velocity values from Odometry Message.
        ## These are in base_footprint frame.
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        self.kf.predict(v, w, dt)  
        self.publish_pose()

    def lidar_callback(self, msg : LaserScan):
        '''Run the Update step of KF each time lidar data is recieved.'''
        ## Convert the LidarScan to PointCloud.
        scan_pcd = self.convert_scan_to_pcd(msg)

        ## Run the ICP on this data.
        icp_result = self.icp_match(scan_pcd)

        if icp_result is not None:
            self.kf.update(icp_result)
            self.publish_pose()

    def convert_scan_to_pcd(self, msg : LaserScan) -> o3d.geometry.PointCloud:
        ## Convert the LaserScan to PointCloud2
        cloud = self.laser_projector.projectLaser(msg)

        pc_data = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)

        ## Convert PointCloud2 data to o3d.PointCloud
        pcd = o3d.geometry.PointCloud()
        points = np.array(list(pc_data))

        ## Translate by x to go to base_footprint
        # translation_x = np.array([-0.064, 0, 0])
        # points = points + translation_x
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def normalize_angle(self, angle):
        '''Normalize the angle between -pi to pi.'''
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def icp_match(self, scan_pcd : o3d.geometry.PointCloud):

        initialization_transform = np.eye(4) if self.initialization_transform is None else self.initialization_transform

        ## Convergence Criteria
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=200
                )

        ## ICP Registration
        reg_p2p = o3d.pipelines.registration.registration_icp(
            scan_pcd, self.map_pcd, 5.0, initialization_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria
        )
        
        transformation = reg_p2p.transformation
        x, y = transformation[0, 3], transformation[1, 3]
        theta = np.arctan2(transformation[1,0], transformation[0, 0])
        theta = self.normalize_angle(theta)

        self.initialization_transform = transformation

        ## TODO: Remove the data coping here.
        scan_pcd_transformed_points = scan_pcd.points
        scan_pcd_transformed = o3d.geometry.PointCloud()
        scan_pcd_transformed.points = scan_pcd_transformed_points
        scan_pcd_transformed.transform(transformation)

        # ## Origin Coordinate Frame
        # # origin_cordi = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # ## Plot the pointcloud for 3D visulaization
        # o3d.visualization.draw_geometries([
        #     scan_pcd,
        #     self.map_pcd,
        #     scan_pcd_transformed,
        #     # origin_cordi,
        # ])

        ## Publish the transformed PointCloud.
        self.transformed_pointcloud_publisher(scan_pcd_transformed)

        return np.array([[x], [y], [theta]])

    def transformed_pointcloud_publisher(self, pcd : o3d.geometry.PointCloud):
        ## Convert the o3d.PointCloud to PointCloud2
        pcd_points = np.asarray(pcd.points)

        ## Convert the points to PointCloud2
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        cloud_msg = point_cloud2.create_cloud(header, fields, pcd_points)
        self.transformed_pointcloud_pub.publish(cloud_msg)
    
    def publish_pose(self):
        '''Publish the localization pose as a TF'''
        x, y, theta = self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'map'
        t.child_frame_id = 'my_localization'#'base_footprint'#
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0
        ## Quaternion
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = np.sin(theta / 2)
        t.transform.rotation.w = np.cos(theta / 2)
        self.br.sendTransform(t)

class RobotController():
    def __init__(self):
        # rospy.init_node('robot_controller')

        ## Transform Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        ## Publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # rospy.spin()

    
    def move_to_waypoints(self, waypoints : list):
        '''Move the robot to the given waypoints.
        waypoints : List of tuples (x, y, theta)'''

        for pose in waypoints:
            self.move_to_pose(pose[0], pose[1], pose[2])

    def move_to_pose(self, x, y, theta):

        vel_msg = Twist()

        while not rospy.is_shutdown():
            ## Get the current pose.
            current_pose = self.tf_buffer.lookup_transform('map', 'my_localization', rospy.Time(0), rospy.Duration(5))

            current_trans = current_pose.transform.translation
            current_rot = current_pose.transform.rotation

            _, _, yaw = tf.transformations.euler_from_quaternion([current_rot.x, current_rot.y, current_rot.z, current_rot.w])

            x_current = current_trans.x
            y_current = current_trans.y
            ## Normalize the angle between -pi to pi.
            theta_current = (yaw + np.pi) % (2 * np.pi) - np.pi

            ## Compute the position errors
            dx = x - x_current
            dy = y - y_current
            distance_to_goal = np.sqrt(dx**2 + dy**2)

            ## Compute the angle error
            angle_to_goal = np.arctan2(dy, dx)
            angle_error = np.arctan2(np.sin(angle_to_goal - theta_current), np.cos(angle_to_goal - theta_current))

            ## Proportional Controller
            Kp_linear = 1.0
            Kp_angular = 1.0
            dist_threshold = 0.05 ## cm
            angle_threshold = 3 * np.pi/180 ## degree
            max_linear_speed = 0.1
            max_angular_speed = 0.5

            rospy.loginfo(f"Current: {x_current:.3f}, {y_current:.3f}, {theta:.3f} | Goal: {x:.3f}, {y:.3f}, {theta:.3f} | Dist: {distance_to_goal:.3f}, Angle: {angle_error:.3f}")

            if distance_to_goal > dist_threshold:
                ## Scale the linear speed based on the angle error.
                linear_speed_scaling_factor = min(1, np.exp(-5*abs(angle_error)))
                vel_msg.linear.x = np.clip(Kp_linear * linear_speed_scaling_factor, 0, max_linear_speed)
                vel_msg.angular.z = np.clip(Kp_angular * angle_error, -max_angular_speed, max_angular_speed)

            else:
                ## Close to the goal.
                vel_msg.linear.x = 0.0

                ## Calculate the angle error.
                angle_error = np.arctan2(np.sin(theta - theta_current), np.cos(theta - theta_current))

                ## Align with the goal orientation.
                if abs(angle_error) > angle_threshold:
                    vel_msg.angular.z = np.clip(Kp_angular * angle_error, -max_angular_speed, max_angular_speed)
                else:
                    rospy.loginfo("Reached the goal. Stopping the robot.")
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(vel_msg)
                    break
            
            self.cmd_vel_pub.publish(vel_msg)

                                            
def main(): 
    ## Start the Localization Node in a separate thread.
    rospy.init_node('my_node')
    localization_thread = threading.Thread(target=LocalizationNode)
    localization_thread.start()

    # # Wait for the Localization Node to start properly.
    # rospy.sleep(5.0)

    # ## Start the Robot Controller
    # controller = RobotController()
    # controller.move_to_waypoints([
    #     (1.60, 0.25, 0.0),
    #     (1.60, 1.17, np.pi/2),
    #     (1.00, 1.17, np.pi + np.pi/15)
    #     ],
    # )

    rospy.spin()
    
if __name__ == '__main__':
    main()