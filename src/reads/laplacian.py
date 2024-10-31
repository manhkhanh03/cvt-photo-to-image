import cv2 as cv 
import numpy as np

img_path = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\imgs\image.png"
img = cv.imread(img_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
laplacian = cv.Laplacian(gray, cv.CV_32F, ksize=3)
laplacian = cv.convertScaleAbs(laplacian)
contours, hierarchy = cv.findContours(laplacian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.imshow("original", img)
cv.waitKey(0)
cv.destroyAllWindows()

