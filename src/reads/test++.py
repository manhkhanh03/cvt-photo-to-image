import easyocr as eo
import cv2 as cv
import threading
import queue
import time
import numpy as np

class VideoOCR:
    def __init__(self, camera_index=1):
        self.reader = eo.Reader(['en', 'vi'])
        self.vid = cv.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(3)
        self.frame_contours = queue.Queue(3)
        self.running = True
        self.current_results = []
        self.text_result = np.ones((1000, 800, 3), np.uint8) * 255
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.reader.readtext(frame)
        if results:
            self.current_results = []
            
            for (bbox, text, score) in results:
                if score > 0.2:                     
                    self.current_results.append(text)

    def ocr_thread(self):
        while self.running:
            if not self.frame_contours.empty():
                frame = self.frame_contours.get()
                self.process_frame(frame)
            # time.sleep(0.001)
            
    def contours(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                laplacian = cv.Laplacian(gray, cv.CV_32F, ksize=3)
                laplacian = cv.convertScaleAbs(laplacian)
                contours, _ = cv.findContours(laplacian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if contours:
                    max_contour = max(contours, key=cv.contourArea)
                    x,y,w, h = cv.boundingRect(max_contour)
                    cropped_img = frame[y:y+h, x:x+w]
                    cv.imshow('crop img', cropped_img)
                    self.frame_contours.put(cropped_img)                
                    key = cv.waitKey(1)
                    if key == ord('q'): 
                        self.running = False
                        break
                    # if not self.frame_contours.full():

    def run(self, name: str):
        contours = threading.Thread(target=self.contours)
        ocr_thread = threading.Thread(target=self.ocr_thread)
        contours.start()
        ocr_thread.start()
        
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
                    line_step = 20
                    for text in self.current_results:
                        font = cv.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        position = line_start
            
                        cv.putText(img_text_result_copy, text, position, font, font_scale, (0, 0, 0), thickness)
                        cv.imshow("Processed", img_text_result_copy)
                        line_start = (line_start[0], line_start[1] + line_step)
                           
                    cv.imshow(name, frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            ocr_thread.join()
            contours.join()
            self.vid.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    ocr = VideoOCR(1)
    name = "Video OCR"
    ocr.run(name)