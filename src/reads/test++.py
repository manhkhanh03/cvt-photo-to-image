import easyocr as eo
import cv2 as cv
import cv2_ext
import threading
import queue
import numpy as np
import time
from PIL import ImageDraw, Image, ImageFont
font = ImageFont.truetype(r"C:\Users\khanh\Documents\cvt-photo-to-text\public\fonts\DejaVuSans.ttf", size=20)

class VideoOCR:
    def __init__(self, camera_index=1):
        self.reader = eo.Reader(['en', 'vi'])
        self.vid = cv.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(3)
        self.frame_contours = queue.Queue(3)
        self.running = True
        self.current_results = []
        self.text_result = np.ones((1000, 800, 3), np.uint8) * 255
        
    def process_frame(self, frame: np.ndarray) -> None:
        results = self.reader.readtext(frame)
        
        self.current_results = []
        if results: 
            for (bbox, text, score) in results:
                if score > 0.2: 
                    self.current_results.append(text)
                    
    def ocr_thread(self) -> None:
        while self.running:
            if not self.frame_contours.empty():
                frame = self.frame_contours.get()
                self.process_frame(frame)
                
            time.sleep(0.1)
            
    def contours(self) -> None:
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                laplacian = cv.Laplacian(gray, cv.CV_32F, ksize=3)
                laplacian = cv.convertScaleAbs(laplacian)
                contours, _ = cv.findContours(laplacian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if contours:
                    max_contour = max(contours, key=cv.contourArea)
                    x, y, w, h = cv.boundingRect(max_contour)
                    print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                    cropped_img = frame[y:y+h, x:x+w]
                    cv.imshow('Crop img', cropped_img)
                    self.frame_contours.put(cropped_img)
                    if cv.waitKey(1) == ord('q'):
                        self.running = False
                        break
            time.sleep(0.1)

    def run(self, name: str) -> None:
        contours_thread = threading.Thread(target=self.contours)
        ocr_thread = threading.Thread(target=self.ocr_thread)
        contours_thread.start()
        ocr_thread.start()
        line_step = 20
        
        try:
            while True:
                ret, frame = self.vid.read()
                img_text_result_copy = self.text_result.copy()
                if not ret:
                    break
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                if self.current_results:
                    line_start = (200, 200)
                    pil_image = Image.fromarray(cv.cvtColor(img_text_result_copy, cv.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    for text in self.current_results:
                        draw.text(line_start, text, (0, 0, 0), font, spacing=2)
                        line_start = (line_start[0], line_start[1] + line_step)
                    
                    img_text_result_copy = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
                    cv.imshow("Processed", img_text_result_copy)
                           
                cv.imshow(name, frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            ocr_thread.join()
            contours_thread.join()
            self.vid.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    ocr = VideoOCR(1)
    ocr.run("Video OCR")