from PIL import Image, ImageFilter
import cv2 as cv

path = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\imgs\IMG_4703.JPG"
# img = Image.open(path)
img = cv.imread(path)

img = img.filter(ImageFilter.GaussianBlur)

img.show()