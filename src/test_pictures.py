import cv2
import numpy as np

import detect_shapes

text_pic_file = 'test/sample2.png'

img = cv2.imread(text_pic_file)
processed_img, counts = detect_shapes.find_shapes(img, debug=True)
detect_shapes.draw_shape_counter(processed_img, *counts)

while True:
    cv2.imshow("processed image", processed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()