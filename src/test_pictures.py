import sys
import cv2
import numpy as np

from detect_shapes import find_shapes, draw_shape_counter

try:
    test_pic = sys.argv[1]
except:
    test_pic = 'test/sample_complex.png'

img = cv2.imread(test_pic)
processed_img, counts = find_shapes(img, debug=True)
draw_shape_counter(processed_img, *counts)

while True:
    cv2.imshow("Shape Detection (Picture)", processed_img)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()