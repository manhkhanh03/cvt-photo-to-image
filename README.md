# Hé lô các bé đến với project hé hé

# Đề tài: Tính năng chuyển chuyển ảnh thành văn bản: Convert photo to text

# Theo như t tìm hiểu trên nền tảng web thì có 4 bước

### Step 1: Đọc ảnh đầu vào

-   Ở bước này các bạn sẽ dùng opencv như đã học ở trên lớp nha
-   Phần này nhóm mình sẽ phát triển để đọc cả photo và video nha. Good luck

### Step 2: Tiền xử lý ảnh

    Ở bước này sẽ có tiến trình như sau:

-   Chuyển ảnh sang ảnh xám
-   Loại bỏ nhiễu(sử dụng bộ lọc Gaussian): như nhiễu tivi
-   Tăng cường độ tương phản
-   Nhị phân hóa ảnh để tách biệt văn bản khỏi nền

### Step 3: Nhận dạng ký tự quang học(OCR):

-   Sử dụng một thư viện OCR như Tesseract. Có thể sử dụng pytesseract

### Step 4: Xử lý hậu kỳ và xuất kết quả:

-   Làm sạch văn bản đầu ra: Ví dụ loại bỏ ký tự đặc biệt không mong muốn
-   Định dạng văn bản(có thể bỏ qua bước này nếu không đủ thời gian)
-   Lưu kết quả vào file hoặc có thể hiện thị thẳng lên màn hình(Bước này sẽ làm sau trước định dạng văn bản để kịp tiến độ đồ án)
