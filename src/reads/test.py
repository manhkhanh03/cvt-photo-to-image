import easyocr
import cv2
 
img_path = r"C:\Users\khanh\Documents\cvt-photo-to-text\public\imgs\text_eng3.jpg"
reader = easyocr.Reader(['vi'])
result = reader.readtext(img_path)
mat = cv2.imread(img_path)

[boxes, texts, scores] = zip(*result)
 
for i, box in enumerate(boxes):    
    top_left     = (int(box[0][0]), int(box[0][1]))
    bottom_right = (int(box[2][0]), int(box[2][1]))
    bottom_left  = (int(box[3][0]), int(box[3][1]))
    font = cv2.FONT_ITALIC
    font_scale = 1
    color = (0, 0, 0)
    position = ((top_left[0] + bottom_left[0]) // 2, (top_left[1] + bottom_left[1]) // 2)
    
    cv2.rectangle(mat, top_left, bottom_right, (249, 245, 245), -1)
    cv2.putText(mat, texts[i], position, font, font_scale, color, 2)

print(texts)
cv2.imshow("result", mat)
cv2.waitKey(0)