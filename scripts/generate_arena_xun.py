import numpy as np
from PIL import Image

# Parameters of the arena
img_name = "arena.pgm"
yaml_name = "map.yaml"
arena_size = 0.92 * 2
resolution = 0.005
img_size = (int)(arena_size / resolution)
origin = [0, 0, 0]
negate = 0
ot = 0.8
ft = 0.1


# Write information to yaml
yaml_f = open(yaml_name, 'w')
yaml_f.write("image: " + img_name + "\n")
yaml_f.write("resolution: " + str(resolution) + "\n")
yaml_f.write("origin: " + str(origin) + "\n")
yaml_f.write("negate: " + str(negate) + "\n")
yaml_f.write("occupied_thresh: " + str(ot) + "\n")
yaml_f.write("free_thresh: " + str(ft) + "\n")

# Write information to the image
arr = np.ones((img_size, img_size)) * 255

for i in range(img_size):
	
	# Two-pixel width line as the border
	for j in range(img_size):
		if i == 0 or i == 1 or i == img_size - 2 or i == img_size - 1:
			arr[i, j] = 0
		else:
			if j == 0 or j == 1 or j == img_size -2  or j == img_size - 1:
				arr[i, j] = 0
arr = (np.asarray(arr)).astype("uint8")
print(arr)
im = Image.fromarray(arr)
im.save(img_name)

