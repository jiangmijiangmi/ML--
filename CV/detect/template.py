import numpy as np
import cv2

img=cv2.imread('faces.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

template = cv2.imread('face_template2.jpg',cv2.IMREAD_GRAYSCALE)

w,h=template.shape[::-1]
method=eval('cv2.TM_CCOEFF')

res=cv2.matchTemplate(gray,template,method)

min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)

top_left=max_loc

bottom_right=(top_left[0]+w,top_left[1]+h)


cv2.rectangle(img,top_left,bottom_right,255,2)

cv2.imshow('cv2.TM_CCOEFF',img)

cv2.waitKey(0)

cv2.destoryAllWindows()