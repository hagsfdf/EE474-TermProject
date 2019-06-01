import numpy as np
import requests
import urllib
import cv2

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

a = 'http://www.shoemarker.co.kr/Upload/ProductImage/010102/8355_1_0180_0180.jpg'
image = url_to_image(a)
print(np.shape(image))

cv2.imshow('image',image)
cv2.waitKey(0)