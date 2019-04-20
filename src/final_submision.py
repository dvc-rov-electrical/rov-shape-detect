import sys

import cv2
import numpy as np

approx_polydp_error = 0.03
triangle_length_thresh = 30
parallel_edge_ratio = 1.5
circle_max_cnt_length = 6
circle_dissimilarity_max = 1.2
circle_angle_range_thresh = 45

# removes redundant wrapper arrays generated with found contours
def unwrap_contour(cnt):
    return np.array([point[0] for point in cnt])

# Calculates the distance between two points
def distance(s1, s2):
    return np.linalg.norm(s1 - s2)

# Calculates the angle between 2 vectors
def angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))

# Calculates the angle between 2 edges of a polygon
def edge_angle(dest1, origin, dest2):
    return angle(dest1 - origin, dest2 - origin)

# Calculates all angles of a polygon, given its vertice locations
def angles_of_shape(points):
    angles = []
    for offset in range(len(points)):
        adj1, origin, adj2 = np.roll(points, offset, axis=0)[:3]
        angles += [edge_angle(adj1, origin, adj2)]
    return np.array(angles)

# Calculates all edge lengths of a polygon, given its vertice locations
def edges_of_shape(points):
    lengths = []
    for offset in range(len(points)):
        p1, p2 = np.roll(points, offset, axis=0)[:2]
        lengths += [np.linalg.norm(p1 - p2)]
    return np.array(lengths)

# Create a contour that represents a perfect circle (used for circle detection)
def perfect_circle_contour(x, y, r, num_points):
    perfect_circle = []
    for theta in range(0, num_points):
        perfect_circle += [[[ r * np.sin(theta) + y, r * np.cos(theta) + x ]]]

    perfect_circle = np.array(perfect_circle, np.int32)
    return perfect_circle

# A BLOB is a Binary Large OBjects, which represents "a group of connected pixels in a binary image".
# This function finds BLOBs from an image and returns a list of dictionaries representing the location,
# size, lower and upper screen coordinates for each BLOB
def find_blobs(image):
    params = cv2.SimpleBlobDetector_Params()

    # Keep "stable" BLOBs by increasing the number of times it appears in the list of generated threshold images
    params.minThreshold = 0
    params.minRepeatability = 4 # higher number yields stabler blobs

    max_min_repeatability = int((params.maxThreshold - params.minThreshold) / params.thresholdStep)
    if params.minRepeatability >= max_min_repeatability:
        raise Exception(
            'Max "minRepeatability" (%s) is exceeded by given minRepeatability (%s)' %
            (max_min_repeatability, params.minRepeatability))

    params.filterByColor = True
    params.blobColor = 0

    # Don't keep shapes that are too big or small
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 20000

    # Don't filter by circularity, since triangles and rectangles aren't very circular
    params.filterByCircularity = False

    # Don't filter by inertia, since triangles, squares, and circles represent regular polygons
    # while rectangles can be very squished and have varying dimensions
    params.filterByInertia = False

    # We don't want concave shapes
    params.filterByConvexity = True
    params.minConvexity = 0.9

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    keypoints = clean_keypoints(keypoints)
    keypoints = keypoint_rect_bounds(keypoints, image.shape)

    return keypoints

# Used for extracting the relevent information from the found BLOBs
def clean_keypoints(keypoints):
    cleaned_keypoints = []
    for keypoint in keypoints:
        cleaned_keypoints += [{
            'center': (np.int(keypoint.pt[0]), np.int(keypoint.pt[1])),
            'size': np.int(keypoint.size / 1.05),
            'lower': None,
            'upper': None
        }]

    return cleaned_keypoints

# Computes the boundaries of a given BLOB, from its KeyPoint data
def get_keypoint_bounds(keypoint, img_shape):
    x, y = keypoint['center']
    sz = keypoint['size']

    lowerY = max(y - sz, 0)
    upperY = min(y + sz, img_shape[0])

    lowerX = max(x - sz, 0)
    upperX = min(x + sz, img_shape[1])

    lower_bound = (lowerX, lowerY)
    upper_bound = (upperX, upperY)
    return lower_bound, upper_bound

# Computes a list of boundary coordinates for a list of BLOBs
def keypoint_rect_bounds(keypoints, img_shape):
    for i in range(len(keypoints)):
        lower, upper = get_keypoint_bounds(keypoints[i], img_shape)
        keypoints[i]['lower'] = lower
        keypoints[i]['upper'] = upper

    return keypoints

def identify_shape(cnt):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    perimeter = cv2.arcLength(cnt, True)
    approx_poly = cv2.approxPolyDP(cnt, approx_polydp_error * perimeter, True)

    # Consider lines as really flat rectangles
    if len(approx_poly) == 2:
        shape = "rectangle"

    # if the shape has 3 vertices, it is a triangle
    elif len(approx_poly) == 3:
        # We want only roughly equilateral triangles, so we'll calculate the edge lengths,
        # get the range of values, and see if it's within a certain threshold
        tri_points = unwrap_contour(approx_poly)
        edge_lengths = edges_of_shape(tri_points)
        edge_length_range = np.max(edge_lengths) - np.min(edge_lengths)

        if edge_length_range <= triangle_length_thresh:
            shape = "triangle"

    # if the shape has 4 vertices, it's a quadrilateral of some kind
    elif len(approx_poly) == 4:
        rect_points = unwrap_contour(approx_poly)
        distances = np.round([np.linalg.norm(point) for point in rect_points], 5)
        min_dist_index = np.where(distances == min(distances))[0][0]
        rotated_points = np.roll(rect_points, min_dist_index, axis=0)

        # We want to label squares and rhombuses as 'squares', rectangles and parallelograms
        # as 'rectangles', and discard kites and quadrilaterals

        # First look for squares and rhombuses
        # Check the max and min edge lengths and make a ratio: a square should have a ratio near 1
        edge_lengths = edges_of_shape(rotated_points)
        edge_length_range = np.max(edge_lengths) / np.min(edge_lengths)

        if edge_length_range <= parallel_edge_ratio:
            shape = "square"
        else:
            # Next, look for rectangles and parallelograms
            # We look at the size of parallel segments of the quadrilateral and expect
            # rectangles and parallelograms to have a edge length ratio between
            # said segments of near 1

            # Extract the specific points in clockwise order
            (nearest_point, middle_point1, farthest_point, middle_point2) = rotated_points

            # Clockwise
            near_cw = (nearest_point, middle_point1)
            far_cw = (farthest_point, middle_point2)
            cw_segments = [near_cw, far_cw]

            # Get distances of 'clockwise' segments
            cw_distances = [distance(*segment) for segment in cw_segments]

            # Get 'square' ratio between parallel segments
            cw_dist_ratio = cw_distances[0] / cw_distances[1]
            if cw_dist_ratio < 1:
                cw_dist_ratio = 1 / cw_dist_ratio

            # Counter-clockwise
            near_ccw = (nearest_point, middle_point2)
            far_ccw = (farthest_point, middle_point1)
            ccw_segments = [near_ccw, far_ccw]

            # Get distances from farthest point to middle 2 points
            ccw_distances = [distance(*segment) for segment in ccw_segments]
            ccw_dist_ratio = ccw_distances[0] / ccw_distances[1]
            if ccw_dist_ratio < 1:
                ccw_dist_ratio = 1 / ccw_dist_ratio

            if cw_dist_ratio < parallel_edge_ratio and ccw_dist_ratio < parallel_edge_ratio:
                shape = "rectangle"

    # otherwise, we assume the shape is a circle or a line (squashed circle)
    elif len(cnt) >= circle_max_cnt_length:
        (x, y), r = cv2.minEnclosingCircle(cnt)

        perfect_circle_cnt = perfect_circle_contour(x, y, r, 45)
        circle_points = unwrap_contour(cnt)

        # Use the Hausdorff Distance to compare the similarity of the found circle contour
        # against a perfect circle's contour
        circle_dissimilarity = cv2.matchShapes(perfect_circle_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0)
        angles = angles_of_shape(circle_points)
        angle_range = np.max(angles) - np.min(angles)

        if angle_range <= circle_angle_range_thresh:
            # Definitely a circle
            shape = "circle"
        elif circle_dissimilarity > circle_dissimilarity_max:
            # Looks more like a line (or a squashed circle)
            shape = "rectangle"

    return shape, approx_poly

def find_shapes(image):
    drawn_image = image.copy()

    # Grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the potential shapes via BLOB detection
    keypoints = find_blobs(gray)

    shape_counts = {
        'circle': 0,
        'triangle': 0,
        'rectangle': 0,
        'square': 0,
    }

    # For each BLOB found...
    for keypoint in keypoints:
        (lowerX, lowerY) = keypoint['lower']
        (upperX, upperY) = keypoint['upper']

        # Crop the BLOB image and apply an inverse threshold, with the said threshold value
        # calculated automatically with Otsu's Method
        roi_image = gray[lowerY:upperY, lowerX:upperX]
        threshed_img = cv2.threshold(roi_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Pull out the possible contours in the ROI and sort them from greatest area to least.
        # The contour with the largest area is the most likely shape that we're looking for.
        shape_cnts = cv2.findContours(threshed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sorted_cnts = sorted(shape_cnts, key=cv2.contourArea, reverse=True)

        if len(sorted_cnts) > 0:
            # There should be only one shape to detect within each ROI image, but sometimes
            # there can be overlapping shapes interfering, hence we only use the largest one
            shape_cnt = sorted_cnts[0]

            # We don't want concave shapes, so find its convex hull, or the set of points that
            # create a polygon that sweeps throughout all the
            convex_hull = cv2.convexHull(shape_cnt)

            # Figure out what shape is there from the convex hull
            shape, shape_hull = identify_shape(convex_hull)

            if shape == "unidentified":
                continue

            shape_counts[shape] += 1

            cv2.drawContours(
                drawn_image,
                [convex_hull + keypoint['center'] - keypoint['size']],
                -1,
                (0, 255, 0),
                2
            )

    return drawn_image, list(shape_counts.values())

# Draws the number of found benthic species found on the lower right hand corner
def draw_shape_counter(input_img, num_circles, num_triangles, num_lines, num_squares):
    final_img = input_img.copy()

    pen_color = (0, 0, 255) # red
    text_options = (cv2.FONT_HERSHEY_SIMPLEX, 1.5, pen_color, 4, cv2.LINE_AA)

    # Draw circle counter
    cv2.circle(
        final_img,
        (final_img.shape[1] - 100, final_img.shape[0] - 200),
        20, pen_color, -1)
    cv2.putText(
        final_img,
        str(num_circles),
        (final_img.shape[1] - 50, final_img.shape[0] - 180),
        *text_options)

    # Draw triangle counter
    side_len = 40
    offsetX = final_img.shape[1] - 120
    offsetY = final_img.shape[0] - 170

    pt1 = (offsetX + side_len // 2	, offsetY)
    pt2 = (offsetX + 0				, offsetY + side_len)
    pt3 = (offsetX + side_len		, offsetY + side_len)
    triangle_cnt = np.array([pt1, pt2, pt3])

    cv2.drawContours(final_img, [triangle_cnt], 0, pen_color, -1)
    cv2.putText(
        final_img,
        str(num_triangles),
        (final_img.shape[1] - 50, final_img.shape[0] - 130),
        *text_options)

    # Draw line counter
    cv2.line(
        final_img,
        (final_img.shape[1] - 120, final_img.shape[0] - 95),
        (final_img.shape[1] - 80, final_img.shape[0] - 95),
        pen_color, 6)
    cv2.putText(
        final_img,
        str(num_lines),
        (final_img.shape[1] - 50, final_img.shape[0] - 80),
        *text_options)

    # Draw square counter
    (originX, originY) = (final_img.shape[1] - 120, final_img.shape[0] - 60)
    side_length = 40

    top_left_coord = (originX, originY)
    bottom_right_coord = (originX + side_length, originY + side_length)

    cv2.rectangle(
        final_img,
        top_left_coord,
        bottom_right_coord,
        pen_color, -1)
    cv2.putText(
        final_img,
        str(num_squares),
        (final_img.shape[1] - 50, final_img.shape[0] - 30),
        *text_options)

    return final_img

if __name__ == "__main__":
    try:
        port_num = int(sys.argv[1])
    except:
        port_num = 0

    cap = cv2.VideoCapture(port_num)

    while True:
        frame = cap.read()[1]

        # Process the image frame and draw the shape counter
        processed_frame, counts = find_shapes(frame)
        final_frame = draw_shape_counter(processed_frame, *counts)

        cv2.imshow("Shape Detection (Webcam)", final_frame)

        # Press ESC to stop the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam feed source
    cap.release()

    # Press ESC to quit
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()