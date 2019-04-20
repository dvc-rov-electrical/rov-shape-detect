import cv2
import numpy as np

from geometry_utils import unwrap_contour, distance, edges_of_shape, angles_of_shape, perfect_circle_contour
from blob_detection import find_blobs, imagify_keypoints
from image_utils import combine_images_vertical

approx_polydp_error = 0.03
triangle_length_thresh = 30
parallel_edge_ratio = 1.5
circle_max_cnt_length = 6
circle_dissimilarity_max = 1.2
circle_angle_range_thresh = 45

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

def find_shapes(image, debug_keypoints=False):
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

    # Draw the ROI images for each blob at the bottom of the camera feed
    if debug_keypoints and len(keypoints) > 0:
        drawn_image = imagify_keypoints(image, keypoints)

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