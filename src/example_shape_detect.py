import detect_shapes
import cv2
import numpy as np

text_pic_file = 'test-pictures/test_pic2.png'

img = cv2.imread(text_pic_file)
processed_img, counts = detect_shapes.find_shapes(img, debug=True)
detect_shapes.draw_shape_counter(processed_img, *counts)

while True:
    cv2.imshow("new image", processed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()