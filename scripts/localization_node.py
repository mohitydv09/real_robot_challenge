#! /usr/bin/env python3

import rospy
import yaml
import tf2_ros
import numpy as np
import open3d as o3d
from PIL import Image
import laser_geometry.laser_geometry as lg

from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped

from kalman_filter import KalmanFilter

class LocalizationNode():
    def __init__(self):
        rospy.loginfo('Localization Node Started.')

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=50)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=10)
        self.transformed_pointcloud_pub = rospy.Publisher('/laser_pointcloud', PointCloud2, queue_size=10)

        self.kf = KalmanFilter()
        self.last_time = None

        ## Load the presaved Pointcloud map
        self.load_map('utils/map.yaml')
        self.map_pcd = o3d.io.read_point_cloud('utils/arena.pcd') 
        self.initialization_transform = None

        ## Helper for LaserScan to PointCloud Conversions.
        self.laser_projector = lg.LaserProjection()

        ## Transform Publisher.
        self.br = tf2_ros.TransformBroadcaster()

        ## Transform Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


    def load_map(self, map_yaml_path):
        '''Load the map data from YAML and PGM files.'''
        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        # Read the PGM file (map image) using PIL
        map_image = Image.open(map_data['image']).convert('L')  # Convert to grayscale
        self.map_pgm = np.array(map_image, dtype=np.uint8)

        self.map_resolution = map_data['resolution']
        self.map_origin = map_data['origin']

        # Create OccupancyGrid message
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        self.map_msg.info.resolution = self.map_resolution
        self.map_msg.info.width = self.map_pgm.shape[1]
        self.map_msg.info.height = self.map_pgm.shape[0]
        self.map_msg.info.origin.position.x = self.map_origin[0]
        self.map_msg.info.origin.position.y = self.map_origin[1]
        self.map_msg.info.origin.position.z = self.map_origin[2]
        self.map_msg.data = self.map_pgm.flatten().tolist()

        # Publish map
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)
        self.publish_map()

    def publish_map(self):
        '''Publish the loaded map as an OccupancyGrid.'''
        self.map_pub.publish(self.map_msg)

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
        # points = points + np.array([-0.064, 0, 0])
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    @staticmethod
    def normalize_angle(angle):
        '''Helper function to Normalize the angle between -pi to pi.'''
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
        t.child_frame_id = 'robot_base'#'base_footprint'#
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0
        ## Quaternion
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = np.sin(theta / 2)
        t.transform.rotation.w = np.cos(theta / 2)
        self.br.sendTransform(t)
        rospy.loginfo(f'Published transform: {t}')

def main():
    rospy.init_node('localization_node')
    LocalizationNode()
    rospy.spin()

if __name__ == '__main__':
    main()