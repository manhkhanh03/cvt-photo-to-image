import cv2 as cv 

img_path = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\imgs\image.png"
img = cv.imread(img_path)
gaussian = cv.GaussianBlur(img, (3,3), 0)
gray = cv.cvtColor(gaussian, cv.COLOR_BGR2GRAY)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
abs_x = cv.convertScaleAbs(grad_x)
abs_y = cv.convertScaleAbs(grad_y)
img_add = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

cv.imshow("original", img)
cv.imshow("laplacian", img_add)
cv.waitKey(0)
cv.destroyAllWindows()