import PIL.Image
image = PIL.Image.open('test.jpg')
import numpy as np

image = np.array(image)
image.shape

print(type (image))
#for i in image:
#	print(i)