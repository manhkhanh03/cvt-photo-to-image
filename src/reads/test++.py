import easyocr as eo
import cv2 as cv
import threading
import queue
import time
import numpy as np

class VideoOCR:
    def __init__(self, camera_index=1):
        self.reader = eo.Reader(['en'])
        self.vid = cv.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(2)
        self.result_queue = queue.Queue()
        self.running = True
        self.current_results = []
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.reader.readtext(frame)
        
        if results:
            processed_frame = frame.copy()
            self.current_results = []
            
            for (bbox, text, score) in results:
                if score > 0.2:                     
                    self.current_results.append((bbox, text, score))
            
            return processed_frame
        return frame

    def ocr_thread(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self.process_frame(frame)
                self.result_queue.put(processed_frame)
            time.sleep(0.001)

    def run(self, name: str):
        ocr_thread = threading.Thread(target=self.ocr_thread)
        ocr_thread.start()

        try:
            while True:
                ret, frame = self.vid.read()
                if not ret:
                    break

                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                if not self.result_queue.empty():
                    processed_frame = self.result_queue.get()
                    cv.imshow(name, processed_frame)
                else:
                    if self.current_results:
                        for (bbox, text, prob) in self.current_results:
                            top_left = (int(bbox[0][0]), int(bbox[0][1]))
                            bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
                            bottom_left = (int(bbox[3][0]), int(bbox[3][1]))
                            
                            font = cv.FONT_ITALIC
                            font_scale = 0.7
                            thickness = 2
                            (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)
                            
                            print(text_width, text_height)
                            position = ((top_left[0] + bottom_left[0]) // 2, (top_left[1] + bottom_left[1]) // 2)
                
                            cv.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)
                            cv.putText(frame, text, position, font, font_scale, (0, 0, 0), thickness)
                           
                    
                    cv.imshow(name, frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            ocr_thread.join()
            self.vid.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    ocr = VideoOCR(1)
    name = "Video OCR"
    ocr.run(name)