import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag

class CaptureArena():
	def __init__(self):
		# Initialization function
		# Just use the default setup, without additional change
		self.pipe = rs.pipeline()
		self.cfg = rs.config()

		self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

		self.pipe.start(self.cfg)

		# The location of the four corners of the arena 
		# in the recorded map (TODO: collected from experiments)
		# Already transposed
		
		# Paper Letter size: 8.5 inches * 11 inches
		paper_w = 8.5 * 0.0254
		paper_h = 11 * 0.0254
		arena_size = 72 * 0.0254
		self.map_corners = np.array([
			[arena_size * 0.5, paper_w * 0.5, 0, 1], 
			[arena_size - 0.5 * paper_h, paper_w * 0.5, 0, 1], 
			[arena_size - 0.5 * paper_h, arena_size - 0.5 * paper_w, 0, 1], 
			[arena_size * 0.5, arena_size - 0.5 * paper_w, 0, 1]])
		
		# The map coordinate of the target object
		self.target_object = np.array([0, 0, 1])

		self.centers = []

	# The function to visualize the result
	def vis(self, tag_family="tag36h11"):
		# Show the image
		while True:
			color_image, arena_corners, target_object = self.grab_rgb_image(tag_family)
			cv2.imshow('rgb', color_image)

			if cv2.waitKey(1) == ord('q'):
				break
		print('-----------------------------------')
		print(tag_family)
		print("Target Object Location in the arena: ")
		print(target_object)
		print("Centers in tags in the image: ")
		print(self.centers)
		print("Map corners: ")
		print(arena_corners)
		print("Transformation: ")
		print(self.K)


	# The function to capture an image and detect the apriltags in it
	def grab_rgb_image(self, tag_family="tag36h11"):
		frame = self.pipe.wait_for_frames()
		color_frame = frame.get_color_frame()

		# Convert the data into a numpy array
		color_image = np.asanyarray(color_frame.get_data())

		# Detect the AprilTag
		gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		options = apriltag.DetectorOptions(families=tag_family)
		detector = apriltag.Detector(options)
		results = detector.detect(gray)

		# Centers of the detected apriltags
		self.centers = []

		# loop over the AprilTag detection results
		r_idx = 0
		arena_corners = np.zeros((4, 2))
		for r in results:
			# extract the bounding box (x, y)-coordinates for the AprilTag
			# and convert each of the (x, y)-coordinate pairs to integers
			(ptA, ptB, ptC, ptD) = r.corners
			ptB = (int(ptB[0]), int(ptB[1]))
			ptC = (int(ptC[0]), int(ptC[1]))
			ptD = (int(ptD[0]), int(ptD[1]))
			ptA = (int(ptA[0]), int(ptA[1]))

			# draw the center (x, y)-coordinates of the AprilTag
			(cX, cY) = (int(r.center[0]), int(r.center[1]))
			cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

			# draw the bounding box of the AprilTag detection
			cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
			cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
			cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
			cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)

			
			# draw the tag family on the image
			tagFamily = r.tag_family.decode("utf-8") + "_" + str(r_idx)
			self.centers.append([cX, cY])
			cv2.putText(color_image, tagFamily, (ptA[0], ptA[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			r_idx += 1

		if (tag_family== "tag36h11" and len(self.centers) == 5):
			# If the tag belongs to type 1
			# the tags represent the corners
			# if (len(centers) < 4):
			# 	return 0 # Not enough samples
			arena_corners = np.array(self.centers)[0:4]
			arena_corners = np.hstack((arena_corners, np.ones((4,1))))

			# TODO: remove this in real experiments!
			# self.map_corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
			# map = K * image
			# K^T = pinv(image^T) * map^T
			
			# arena_corners & map_corners are already transposed
			self.K = np.transpose(np.matmul(
					np.linalg.pinv(arena_corners),
					self.map_corners
				)
			)

			self.target_object = np.matmul(self.K, 
				np.transpose(np.array([self.centers[-1][0], self.centers[-1][1], 1]))
				)

		

		return color_image, arena_corners, self.target_object


	def close(self):
		self.pipe.stop()

def main():
	capture_area = CaptureArena()
	# Grab an image to obtain the transformation
	capture_area.vis(tag_family="tag36h11")


	capture_area.close()
	


if __name__ == "__main__":
    main()
