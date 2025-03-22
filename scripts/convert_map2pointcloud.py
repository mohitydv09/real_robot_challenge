import cv2
import yaml
import numpy as np
import open3d as o3d


## Load the map files
yaml_file_path = 'map.yaml'
pgm_file_path = 'arena.pgm'

## Load the yaml data
with open(yaml_file_path, 'r') as f:
    map_metadata = yaml.safe_load(f)

map_image = cv2.imread(pgm_file_path, cv2.IMREAD_GRAYSCALE)
print("Map image size: ", map_image.shape)

resolution = map_metadata['resolution']
origin = map_metadata['origin']

point_cloud = []

x_min = 0
x_max = map_image.shape[0] * resolution
y_min = 0
y_max = map_image.shape[1] * resolution

num_points_x = int((x_max - x_min) / resolution)
num_points_y = int((y_max - y_min) / resolution)

## Generate horizontal lines.
x_line = np.linspace(x_min, x_max, num_points_x)
y_line = np.linspace(y_min, y_max, num_points_y)
x_zeros = np.zeros_like(x_line)
x_max_line = np.ones_like(x_line) * x_max
y_zeros = np.zeros_like(y_line)
y_max_line = np.ones_like(y_line) * y_max

## Form a Corner
print("Num Points x: ", num_points_x)
x_new = np.linspace(1.84-0.4, 1.84, 400)
y_new1 = np.ones_like(x_new) * 0.5

y_new = np.linspace(0, 0.5, 500)
x_new1 = np.ones_like(y_new) * (1.84-0.4)

corner1 = np.vstack((x_new, y_new1))
corner2 = np.vstack((x_new1, y_new))

## Form the center thing.
x_center_corner_hor = np.linspace(0, 0.4, 400)


y_center_corner_hor_1 = np.ones_like(x_center_corner_hor) * 0.67
y_center_corner_hor_2 = np.ones_like(x_center_corner_hor) * (0.67 + 0.5)

y_center_corner_ver = np.linspace(0.67, 0.67+0.5, 500)
x_center_corner_ver = np.ones_like(y_center_corner_ver) * 0.4

first_hor_line = np.vstack((x_center_corner_hor, y_center_corner_hor_1))
second_hor_line = np.vstack((x_center_corner_hor, y_center_corner_hor_2))
center_verticle_line = np.vstack((x_center_corner_ver, y_center_corner_ver))

## Form the points.
x_axis = np.vstack((x_line, y_zeros))
x_max_line = np.vstack((x_max_line, y_line))
y_axis = np.vstack((x_zeros, y_line))
y_max_line = np.vstack((x_line, y_max_line))

points = np.hstack((x_axis, x_max_line, y_axis, y_max_line, corner1, corner2, first_hor_line, second_hor_line, center_verticle_line))
z_values = np.zeros((points.shape[1]))

points = np.vstack((points, z_values))

print("PointCloud Shape ", points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.T)

## Save the PointCloud for later use
o3d.io.write_point_cloud('arena.pcd', pcd)