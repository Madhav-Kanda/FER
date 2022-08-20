import cv2
import numpy as np
import skimage.exposure as exposure
import os



inPath="train"
outPath="ThreeChannels"

for folders in os.listdir(inPath):
    print(folders)
    folderPath=os.path.join(inPath,folders)
    for imagePath in os.listdir(folderPath):
        inputPath = os.path.join(folderPath, imagePath)
        img = cv2.imread(inputPath)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
        laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
        laplacian = exposure.rescale_intensity(laplacian, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
        # cv2.imshow('image',laplacian)
        # cv2.waitKey(0)
        fullOutPath = os.path.join(outPath,folderPath)
        fullOutPath=os.path.join(fullOutPath,imagePath)
        # print(fullOutPath)
        cv2.imwrite(fullOutPath, laplacian)
        cv2.waitKey(0)