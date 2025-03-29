import numpy as np
import open3d as o3d
from PIL import Image

# Parameters of the arena
img_name = "arena.pgm"
yaml_name = "map.yaml"
pcd_name = 'arena.pcd'
arena_size = 0.92 * 2
resolution = 0.005
img_size = (int)(arena_size / resolution) ## 368
origin = [0, 0, 0]
negate = 0
occupancy_threshold = 0.8
free_threshold = 0.1

def point_to_pixel(point):
	## Convert the point to pixel
	x = int((point[0] - origin[0]) / resolution)
	y = int((point[1] - origin[1]) / resolution)
	y = img_size - y - 1

	x = max(0, min(x, img_size - 1))
	y = max(0, min(y, img_size - 1))
	return (x, y)

def draw_line(arr, points, start, end):

	x0, y0 = point_to_pixel(start)
	x1, y1 = point_to_pixel(end)

	# Bresenham's line algorithm
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)

	## Step Direction
	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1
	err = dx - dy

	while True:
		## Swap the coordinates for image indexing
		arr[y0, x0] = 0

		## Add point to the points list
		world_x = x0 * resolution + origin[0]
		world_y = (img_size - y0 - 1) * resolution + origin[1]
		points.append([world_x, world_y])

		## End of the line
		if x0 == x1 and y0 == y1:
			break
		
		## Calculate the next point
		e2 = 2 * err
		if e2 > -dy:
			err -= dy
			x0 += sx
		if e2 < dx:
			err += dx
			y0 += sy

# Write information to yaml
with open(yaml_name, 'w') as yaml_f:
	yaml_f.write("image: " + 'utils/' + img_name + "\n")
	yaml_f.write("resolution: " + str(resolution) + "\n")
	yaml_f.write("origin: " + str(origin) + "\n")
	yaml_f.write("negate: " + str(negate) + "\n")
	yaml_f.write("occupied_thresh: " + str(occupancy_threshold) + "\n")
	yaml_f.write("free_thresh: " + str(free_threshold) + "\n")

# Create a blank image
arr = np.ones((img_size, img_size)) * 255
points = []

## Draw the required lines on the image

draw_line(arr, points, (0, 0), (0, arena_size))
draw_line(arr, points, (0, 0), (arena_size, 0))
draw_line(arr, points, (arena_size, 0), (arena_size, arena_size))
draw_line(arr, points, (0, arena_size), (arena_size, arena_size))

# draw_line(arr, points, (0.48, 0.51), (1.37, 0.51))
# draw_line(arr, points, (0.48, 0.51), (0.48, 1.42))
# draw_line(arr, points, (0.48, 1.42), (1.37, 1.42))
draw_line(arr, points, (0 , 0.92), (0.92, 0.92))

## Write the image to a file
arr = arr.astype(np.uint8)
im = Image.fromarray(arr)
im.save(img_name)
print("Image saved to ", img_name)

## Convert points to point cloud
points = np.array(points)
z_values = np.zeros((points.shape[0], 1))
points = np.hstack((points, z_values))

## Write the points to a pcd file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

## Visualize the point cloud with coordinate frame for debugging
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, coordinate_frame])

## Save the PointCloud for later use
o3d.io.write_point_cloud(pcd_name, pcd)
print("Point Cloud saved to ", pcd_name)