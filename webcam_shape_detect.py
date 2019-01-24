import detect_shapes
import cv2
import numpy as np

cap = cv2.VideoCapture(0+1)

while True:
	ret, frame = cap.read()
	#frame = cv2.imread('shapes_and_colors.png')
	newframe = detect_shapes.find_shape(frame)
	cv2.imshow("frame", newframe)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	 	break

cap.release()
cv2.destroyAllWindows()