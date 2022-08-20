import cv2
import numpy as np
import skimage.exposure as exposure
import os



inPath="test"
outPath="ThreeChannels"

for folders in os.listdir(inPath):
    print(folders)
    folderPath=os.path.join(inPath,folders)
    for imagePath in os.listdir(folderPath):
        inputPath = os.path.join(folderPath, imagePath)
        img = cv2.imread(inputPath)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3) 
        sobelx_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
        sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
        sobelx2 = cv2.multiply(sobelx,sobelx)
        sobely2 = cv2.multiply(sobely,sobely)
        sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
        sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
        fullOutPath = os.path.join(outPath,folderPath)
        fullOutPath=os.path.join(fullOutPath,imagePath)
        # print(fullOutPath)
        cv2.imwrite(fullOutPath, sobel_magnitude)
        cv2.waitKey(0)