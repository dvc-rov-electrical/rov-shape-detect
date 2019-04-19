import cv2
import numpy as np

import detect_shapes

cap = cv2.VideoCapture(4 - 4)

while True:
	print("\n\n========= REAL START =========\n\n")
	frame = cap.read()[1]
	processed_frame, counts = detect_shapes.find_shapes(frame, debug=True)
	# detect_shapes.draw_shape_counter(processed_frame, *counts)
	cv2.imshow("frame", processed_frame)
	print("\n\n========= REAL END =========\n\n")

	if cv2.waitKey(1) & 0xFF == ord('q'):
	 	break

cap.release()

if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()