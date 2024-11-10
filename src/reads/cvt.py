import hashlib
import easyocr as eo
import cv2 as cv
import cv2_ext
import threading
import queue
import numpy as np
import time
from PIL import ImageDraw, Image, ImageFont
from screeninfo import get_monitors
font = ImageFont.truetype(
    r"C:\Users\khanh\Documents\cvt-photo-to-text\public\fonts\DejaVuSans.ttf", size=20)


class VideoOCR:
    def __init__(self, camera_index=1):
        self.reader = eo.Reader(['en'])
        self.vid = cv.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(3)
        self.running = True
        self.current_results = []
        self.current_crop = None
        self.text_result = np.ones((1200, 980, 1), np.uint8) * 255

    def process_frame(self, frame: np.ndarray) -> None:
        results = self.reader.readtext(frame)

        self.current_results = []
        if results:
            for (bbox, text, score) in results:
                if score > 0.2:
                    self.current_results.append(text)

    def ocr_thread(self) -> None:
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.process_frame(frame)

            time.sleep(0.1)

    def contours_process(self, frame: np.ndarray) -> None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gaussian = cv.GaussianBlur(gray, (3, 3), 0)
        _, threshold = cv.threshold(
            gaussian, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        contours, _ = cv.findContours(
            threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if contours:
            max_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(max_contour)
            cropped_img = frame[y:y + h, x:x + w]
            return [cropped_img, x, y, w, h]

        return [None, None, None, None, None]

    def image_hash(self, image: np.ndarray) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()

    def run(self, name: str) -> None:
        ocr_thread = threading.Thread(target=self.ocr_thread)
        ocr_thread.start()
        line_step = 20
        x_prev, y_prev, w_prev, h_prev = 0, 0, 0, 0
        hash_prev = ""

        try:
            while True:
                ret, frame = self.vid.read()
                img_text_result_copy = self.text_result.copy()
                if not ret:
                    break

                if not self.frame_queue.full():
                    frame_processed, x, y, w, h = self.contours_process(frame.copy())
                    cv.imshow("Cropped", frame_processed)
                    hash_current = self.image_hash(frame_processed)
                    if x != x_prev or y != y_prev or w != w_prev or h != h_prev or hash_current != hash_prev:
                        self.frame_queue.put(frame_processed)
                        x_prev, y_prev, w_prev, h_prev = x, y, w, h
                        hash_prev = hash_current

                if self.current_results:
                    line_start = (200, 200)
                    pil_image = Image.fromarray(cv.cvtColor(
                        img_text_result_copy, cv.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    for text in self.current_results:
                        draw.text(line_start, text, (0, 0, 0), font, spacing=2)
                        line_start = (line_start[0], line_start[1] + line_step)

                    img_text_result_copy = cv.cvtColor(
                        np.array(pil_image), cv.COLOR_RGB2BGR)
                    cv.imshow("Processed", img_text_result_copy)

                cv.imshow(name, frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            ocr_thread.join()
            self.vid.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    name = "Video OCR"
    ocr = VideoOCR(1)
    ocr.run(name)
