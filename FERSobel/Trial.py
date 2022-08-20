import cv2
import numpy as np
import os
import imutils
from PIL import Image as im
from matplotlib import pyplot as plt


# INPUT_IMAGE="FER_laplacian/img_conv.jpg"
# OUTPUT_IMAGE="FER_laplacian/test.jpg"



inputPath="photo2.png"
# outPath="test/ThreeChannels/angry"

# for imagePath in os.listdir(inPath):
    # print(imagePath)
    # inputPath = os.path.join(inPath, imagePath)
    # print(inputPath)
# img = cv2.imread(inputPath)
matrix=[]
for i in range(256):
    lst=[i]*256
    matrix.append(lst)
# print(matrix)

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
edge_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
# cv2.imshow('image',edge_x)
# cv2.waitKey(0)
edge_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# cv2.imshow('image',edge_y)
# cv2.waitKey(0)    
finalmatrix=np.sqrt(np.square(edge_x)+np.square(edge_y))
# print(finalmatrix)
# print(np.amax(finalmatrix))
finalmatrix=(finalmatrix/np.amax(finalmatrix))
# print(finalmatrix[:][1])
finalmatrix=finalmatrix*255
print(finalmatrix[:][1])
print(finalmatrix.dtype)
finalmatrix=np.trunc(finalmatrix)
print(finalmatrix[:][1])
# print(finalmatrix.dtype)
cv2.imshow('image',finalmatrix)
cv2.waitKey(0) 
img=np.stack([img,edge_x,edge_y])
img=img.T
# cv2.imshow('image',img)
# cv2.waitKey(0) 
# print(img)
rotated=imutils.rotate_bound(img,90)
cv2.imshow('image',rotated)
cv2.waitKey(0) 
# print(rotated)
# fullOutPath = os.path.join(outPath,imagePath)
# cv2.imwrite(fullOutPath, rotated)
cv2.waitKey(0)
# print(fullOutPath)