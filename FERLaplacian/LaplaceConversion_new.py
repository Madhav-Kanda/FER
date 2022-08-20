import cv2
import numpy as np
import os

# INPUT_IMAGE="FER_laplacian/img_conv.jpg"
# OUTPUT_IMAGE="FER_laplacian/test.jpg"

inPath="FER_Sobel/train/sad"
outPath="FER_Sobel/train/ThreeChannels/sad"

for imagePath in os.listdir(inPath):
    # print(imagePath)
    inputPath = os.path.join(inPath, imagePath)
    # print(inputPath)
    img = cv2.imread(inputPath)
    # cv2.imshow("image",img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
    # edge_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    # edge_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)    
    laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
    img=np.stack([img,laplacian])
    img=img.T
    # print(img.shape)
    # print(img)
    fullOutPath = os.path.join(outPath,imagePath)
    cv2.imwrite(fullOutPath, img)
    cv2.waitKey(0)
    # print(fullOutPath)