import easyocr
import cv2
import threading
import queue
import time
import numpy as np

class VideoOCR:
    def __init__(self, camera_index=1):
        self.reader = easyocr.Reader(['en'])
        self.vid = cv2.VideoCapture(camera_index)
        self.frame_queue = queue.Queue(2)
        self.result_queue = queue.Queue()
        self.running = True
        self.current_results = []  # Lưu kết quả hiện tại
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Thực hiện OCR trực tiếp trên frame gốc
        results = self.reader.readtext(frame)
        
        if results:
            # Xử lý kết quả OCR
            processed_frame = frame.copy()
            self.current_results = []  # Reset kết quả cũ
            
            for (bbox, text, prob) in results:
                if prob > 0.2:  # Lọc kết quả có độ tin cậy > 0.2
                    # Vẽ bbox và text
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    
                    # Lưu kết quả để hiển thị liên tục
                    self.current_results.append((bbox, text, prob))
                    
                    # Vẽ hộp chứa text với màu xanh nhạt
                    cv2.rectangle(processed_frame, top_left, bottom_right, (0, 255, 0), 2)
                    
                    # Tạo background trắng cho text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Tạo background
                    padding = 5
                    bg_width = text_width + 2 * padding
                    bg_height = text_height + 2 * padding
                    
                    # Tính vị trí cho background
                    bg_y = top_left[1] - bg_height - 5
                    if bg_y < 0:  # Nếu vượt quá phía trên, đặt text bên dưới bbox
                        bg_y = bottom_right[1] + 5
                        
                    bg_x = top_left[0]
                    if bg_x + bg_width > processed_frame.shape[1]:
                        bg_x = processed_frame.shape[1] - bg_width
                    if bg_x < 0:
                        bg_x = 0
                        
                    # Vẽ background và text
                    try:
                        # Tạo background màu trắng
                        background = np.full((bg_height, bg_width, 3), (255, 255, 255), dtype=np.uint8)
                        # Vẽ text lên background
                        cv2.putText(background, text, 
                                  (padding, text_height + padding // 2),
                                  font, font_scale, (0, 0, 0), thickness)
                        # Chèn background vào frame
                        processed_frame[bg_y:bg_y+bg_height, bg_x:bg_x+bg_width] = background
                    except ValueError:
                        continue
            
            return processed_frame
        return frame

    def ocr_thread(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self.process_frame(frame)
                self.result_queue.put(processed_frame)
            time.sleep(0.001)  # Giảm thời gian sleep để xử lý nhanh hơn

    def run(self):
        # Khởi tạo thread xử lý OCR
        ocr_thread = threading.Thread(target=self.ocr_thread)
        ocr_thread.start()

        try:
            while True:
                ret, frame = self.vid.read()
                if not ret:
                    break

                # Đưa mỗi frame vào queue để xử lý
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                # Hiển thị frame đã xử lý OCR nếu có
                if not self.result_queue.empty():
                    processed_frame = self.result_queue.get()
                    cv2.imshow('Video OCR', processed_frame)
                else:
                    # Vẽ kết quả cũ lên frame hiện tại nếu có
                    if self.current_results:
                        for (bbox, text, prob) in self.current_results:
                            (top_left, _, bottom_right, _) = bbox
                            top_left = tuple(map(int, top_left))
                            bottom_right = tuple(map(int, bottom_right))
                            
                            # # Vẽ lại hộp và text
                            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                            cv2.putText(frame, text, 
                                      (top_left[0], top_left[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Video OCR', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.running = False
            ocr_thread.join()
            self.vid.release()
            cv2.destroyAllWindows()

# Chạy chương trình
if __name__ == "__main__":
    ocr = VideoOCR(1)
    ocr.run()