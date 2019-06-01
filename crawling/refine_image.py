import numpy as np
import requests
import urllib
import cv2


def refine_image(url, dirIm):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    size = np.shape(image)
    
    if (size[0] > 180 or size[1] > 180):
        dst = cv2.resize(image, dsize=(180, 180), interpolation=cv2.INTER_AREA)
    elif (size[0] < 180 or size[1] < 180):
        dst = cv2.resize(image, dsize=(180, 180), interpolation=cv2.INTER_CUBIC)
	# return the image
    else:
        dst = image
    
    cv2.imwrite(dirIm, dst)
