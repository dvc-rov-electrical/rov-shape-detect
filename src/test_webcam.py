import cv2
import numpy as np

from detect_shapes import find_shapes, draw_shape_counter

cap = cv2.VideoCapture(4 - 0)

while True:
	print("\n========= REAL START =========\n")
	frame = cap.read()[1]
	processed_frame, counts = find_shapes(frame, debug=True)
	draw_shape_counter(processed_frame, *counts)
	cv2.imshow("Shape Detection", processed_frame)
	print("\n========= REAL END =========\n")

	# Press ESC to stop the webcam feed
	if cv2.waitKey(1) & 0xFF == 27:
	 	break

# Release the webcam feed source
cap.release()

# Press ESC to quit
if cv2.waitKey(0) & 0xFF == 27:
	cv2.destroyAllWindows()