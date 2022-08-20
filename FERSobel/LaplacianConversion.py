import cv2
import numpy as np
import os
import imutils
from PIL import Image as im
from matplotlib import pyplot as plt
import skimage.exposure as exposure



# INPUT_IMAGE="FER_laplacian/img_conv.jpg"
# OUTPUT_IMAGE="FER_laplacian/test.jpg"

inputPath="photo1.jpg"


# img=matrix
# img = im.fromarray(matrix)
# print(matrix.dtype)
img=cv2.imread(inputPath)
# print(img.dtype)
# plt.imshow(matrix, interpolation='nearest')
# plt.show()
cv2.imshow('image',img)
cv2.waitKey(0)
# cv2.imshow("image",img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
# print(img[:][1])
laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
laplacian_magnitude = exposure.rescale_intensity(laplacian, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

# cv2.imshow('image',edge_x)
# cv2.waitKey(0)
# edge_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# cv2.imshow('image',edge_y)
# cv2.waitKey(0)    
# finalmatrix=np.sqrt(np.square(edge_x)+np.square(edge_y))
# print(finalmatrix)
# print(np.amax(finalmatrix))
# finalmatrix=(finalmatrix/np.amax(finalmatrix))
# print(finalmatrix[:][1])
# finalmatrix=finalmatrix*255
# print(finalmatrix[:][1])
# print(finalmatrix.dtype)
# finalmatrix=np.trunc(finalmatrix)
# print(finalmatrix[:][1])
# # print(finalmatrix.dtype)
# cv2.imshow('image',finalmatrix)
# cv2.waitKey(0) 
# img=np.stack([img,edge_x,edge_y])
# img=img.T
# cv2.imshow('image',img)
# cv2.waitKey(0) 
# print(img)
# rotated=imutils.rotate_bound(img,90)
cv2.imshow('image',laplacian)
cv2.imshow('image',laplacian_magnitude)
cv2.waitKey(0) 
# print(rotated)
# fullOutPath = os.path.join(outPath,imagePath)
# cv2.imwrite(fullOutPath, rotated)
cv2.waitKey(0)
# print(fullOutPath)