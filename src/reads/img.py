import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ImageToText: 
    def __init__(self, img):
        self.img = img
        self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
    def image_to_text(self):    
        text = pytesseract.image_to_string(
            self.gray, lang='eng'
        )
        print(text)
        return text
        # return ' '.join(text.split())
    
class Video: 
    def __init__(self, path = None) -> None:
        if path: 
            self.video = cv.VideoCapture(path)
        else: 
            self.video = cv.VideoCapture(0)
        
    def read_img(self, win_name):
        while True:
            ret, frame = self.video.read()
            if not ret: break
            cv.imshow(win_name, frame)
            img = ImageToText(frame)
            text = img.image_to_text()
            print(text)
                        
            key = cv.waitKey(30)
            if key == ord('q'): 
                self.video.release()
                break
            
            

path = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\imgs\i1Abv.png"
pathVideo = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\videos\text-eng.mp4"
img = cv.imread(path)
img = ImageToText(img)

print(img.image_to_text())
# video = Video(pathVideo)
# video.read_img('Video')
cv.destroyAllWindows()