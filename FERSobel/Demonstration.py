import cv2
import numpy as np
import skimage.exposure as exposure

# read the image
img = cv2.imread('photo1.jpg')

# convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# blur
# blur = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)

# apply sobel derivatives
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
# cv2.imshow('sobelx', sobelx) 
# cv2.waitKey(0)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
# cv2.imshow('sobely', sobely) 
# cv2.waitKey(0)

# optionally normalize to range 0 to 255 for proper display
sobelx_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

# square 
sobelx2 = cv2.multiply(sobelx,sobelx)
sobely2 = cv2.multiply(sobely,sobely)

sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
# cv2.imshow('sobel_magnitude', sobel_magnitude) 
# cv2.waitKey(0)

sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

cv2.imshow('sobel_magnitude', sobel_magnitude)  
cv2.waitKey(0)
cv2.destroyAllWindows()
