import cv2
from matplotlib import pyplot as plt

img=cv2.imread('butterfly.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()
kp=sift.detect(gray,None)
img1=cv2.drawKeypoints(gray,kp,None)


surf = cv2.xfeatures2d.SURF_create(400)
surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(gray,kp,None)


orb=cv2.ORB_create()
kp=orb.detect(gray,None)
img3=cv2.drawKeypoints(gray,kp,None)


plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(img1)
plt.subplot(2,2,3)
plt.imshow(img2)
plt.subplot(2,2,4)
plt.imshow(img3)
plt.show()