# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
import imutils
import cv2

def identify_shape(cnt):
	# initialize the shape name and approximate the contour
	shape = "unidentified"
	peri = cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

	# if the shape is a triangle, it will have 3 vertices
	if len(approx) == 3:
		shape = "triangle"

	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	elif len(approx) == 4:
		# compute the bounding box of the contour and use the
		# bounding box to compute the aspect ratio
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)

		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

	# otherwise, we assume the shape is a circle
	else:
		shape = "circle"

	# return the name of the shape
	return shape

def find_shape(image):
	# resized = imutils.resize(image, width=500)
	#ratio = image.shape[0] / float(resized.shape[0])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)[1]
	# return thresh

	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#print([cv2.contourArea(c) for c in cnts])
	# loop over the contours
	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		area = cv2.contourArea(c)
		if area > 20000 or area < 100:
			continue
		
		M = cv2.moments(c)

		if M["m00"] == 0:
			continue
		
		cX = int((M["m10"] / M["m00"]) * 1)
		cY = int((M["m01"] / M["m00"]) * 1)
		
		shape = identify_shape(c)
		if shape == "unidentified":
			continue

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = c.astype("float")
		#c *= ratio
		c = c.astype("int")
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 255, 255), 2)

	return image