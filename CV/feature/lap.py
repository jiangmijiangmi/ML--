import numpy as np
from PIL import Image
img=Image.open('butterfly.jpg')
img.show()

img_gray=img.convert('L')
img_gray.show()

img_arr=np.array(img_gray)
h,w=img_arr.shape
new_img=np.zeros((h,w))

for i in range (2,h-1):
    for j in range (2,w-1):
        new_img[i][j]=img_arr[i+1,j]+img_arr[i-1,j]+img_arr[i,j+1]+img_arr[i,j-1]-4*img_arr[i,j]

laplace_img=new_img+img_arr

img_laplace = Image.fromarray(np.uint8(new_img))
img_laplace.show()


img_laplace2 = Image.fromarray(np.uint8(laplace_img))
img_laplace2.show()