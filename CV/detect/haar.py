import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

img = cv2.imread('faces.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.2,8)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    eye_gray=gray[y:y+h,x:x+w]
    eye_color=img[y:y+h,x:x+w]

    eyes=eye_cascade.detectMultiScale(eye_gray)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.namedWindow('faces',cv2.WINDOW_AUTOSIZE)
cv2.imshow('face',img)

cv2.waitKey(0)
cv2.destroyAllWindows()