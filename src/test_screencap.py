import sys
import cv2
import numpy as np
from mss import mss

from detect_shapes import find_shapes, draw_shape_counter


monitor = {"top": 0, "left": 0, "width": 960, "height": 1080}

with mss() as sct:
    while True:
        screen = np.array(sct.grab(monitor))

        # Process the image frame and draw the shape counter
        processed_screen, counts = find_shapes(screen)
        final_screen = draw_shape_counter(processed_screen, *counts)

        cv2.imshow("Shape Detection (Screen Capture)", final_screen)

<<<<<<< HEAD
        # Press ESC to stop the screen feed
=======
        # Press ESC to stop the webcam feed
>>>>>>> dc349a6... added shape detection via screen sharing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Press ESC to quit
if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()