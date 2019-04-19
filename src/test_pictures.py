import cv2
import numpy as np

from detect_shapes import find_shapes, draw_shape_counter

test_pic = 'test/sample2.png'

img = cv2.imread(test_pic)
processed_img, counts = find_shapes(img, debug=True)
draw_shape_counter(processed_img, *counts)

while True:
    cv2.imshow("Shape Detection (Picture)", processed_img)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()