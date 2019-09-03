import numpy as np
import cv2
import matplotlib.pyplot as plt

box=cv2.imread('box.png',cv2.COLOR_BGR2GRAY)
box2=cv2.imread('box_in_scene.png',cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(box,None)
kp2,des2=sift.detectAndCompute(box2,None)

bf = cv2.BFMatcher()
match=bf.match(des1,des2)
matches = sorted(match, key = lambda x:x.distance)
img1 = cv2.drawMatches(box,kp1,box2,kp2,matches[:20],None, flags=2)

surf=cv2.xfeatures2d.SURF_create(400)
kp1,des1=sift.detectAndCompute(box,None)
kp2,des2=sift.detectAndCompute(box2,None)
bf = cv2.BFMatcher()
match=bf.match(des1,des2)
matches = sorted(match, key = lambda x:x.distance)
img2 = cv2.drawMatches(box,kp1,box2,kp2,matches[:20],None, flags=2)




orb=cv2.ORB_create()
kp1,des1=orb.detectAndCompute(box,None)
kp2,des2=orb.detectAndCompute(box2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
match=bf.match(des1,des2)
matches = sorted(match, key = lambda x:x.distance)
img3= cv2.drawMatches(box,kp1,box2,kp2,matches[:20],None, flags=2)

plt.subplot(3,1,1)
plt.imshow(img1)
plt.subplot(3,1,2)
plt.imshow(img2)
plt.subplot(3,1,3)
plt.imshow(img3)
plt.show()