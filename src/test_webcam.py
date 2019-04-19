import sys
import cv2
import numpy as np

from detect_shapes import find_shapes, draw_shape_counter

try:
    port_num = int(sys.argv[1])
except:
    port_num = 0

cap = cv2.VideoCapture(port_num)

while True:
	print("\n========= START =========\n")
	frame = cap.read()[1]
	processed_frame, counts = find_shapes(frame, debug=True)
	final_frame = draw_shape_counter(processed_frame, *counts)
	cv2.imshow("Shape Detection (Webcam)", final_frame)
	print("\n========= END =========\n")

	# Press ESC to stop the webcam feed
	if cv2.waitKey(1) & 0xFF == ord('q'):
	 	break

# Release the webcam feed source
cap.release()

# Press ESC to quit
if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()