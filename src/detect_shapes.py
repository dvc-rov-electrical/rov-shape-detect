import cv2
import numpy as np

from geometry_utils import unwrap_contour, edge_lengths_of_shape, perfect_circle_contour
from blob_detection import find_blobs, imagify_keypoints
from image_utils import combine_images_vertical

min_area = 200
max_area = 20000

approx_polydp_error = 0.03
circle_max_cnt_length = 6
triangle_length_thresh = 30
square_aspect_ratio_thresh = 1.5
circle_dissimilarity_max = 1.2

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

        edge_lengths = edge_lengths_of_shape(tri_points)
        edge_length_range = np.max(edge_lengths) - np.min(edge_lengths)
        print('edge length range (triangle):', edge_length_range)

        if edge_length_range <= triangle_length_thresh:
            shape = "triangle"

    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(approx_poly) == 4:
        rect_points = unwrap_contour(approx_poly)
        distances = np.round([np.linalg.norm(point) for point in rect_points], 5)
        min_dist_index = np.where(distances == min(distances))[0][0]
        rotated_points = np.roll(rect_points, min_dist_index, axis=0)

        # Get nearest/farthest points from origin
        (nearest_point, middle_point1, farthest_point, middle_point2) = rotated_points

        # Get distances from nearest point to middle 2 points
        midnear_segments = [(nearest_point, middle_point1), (nearest_point, middle_point2)]
        midnear_distances = [np.linalg.norm(p1 - p2) for p1, p2 in midnear_segments]
        midnear_dist_ratio = midnear_distances[0] / midnear_distances[1]
        if midnear_dist_ratio < 1:
            midnear_dist_ratio = 1 / midnear_dist_ratio

        # Get distances from farthest point to middle 2 points
        midfar_segments = [(farthest_point, middle_point1), (farthest_point, middle_point2)]
        midfar_distances = [np.linalg.norm(p1 - p2) for p1, p2 in midfar_segments]
        midfar_dist_ratio = midfar_distances[0] / midfar_distances[1]
        if midfar_dist_ratio < 1:
            midfar_dist_ratio = 1 / midfar_dist_ratio

        print('%.2f %.2f' % (midnear_dist_ratio, midfar_dist_ratio))
        ratio_diff = np.abs(midnear_dist_ratio / midfar_dist_ratio)
        if ratio_diff < 1:
            ratio_diff = 1 / ratio_diff

        if (ratio_diff < 1.5):
            shape = "square" if midnear_dist_ratio < square_aspect_ratio_thresh and midfar_dist_ratio < square_aspect_ratio_thresh else "rectangle"

    # otherwise, we assume the shape is a circle
    elif len(cnt) >= circle_max_cnt_length:
        (x, y), r = cv2.minEnclosingCircle(cnt)

        perfect_circle_cnt = perfect_circle_contour(x, y, r, 45)
        circle_points = unwrap_contour(cnt)

        circle_dissimilarity = cv2.matchShapes(perfect_circle_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0)
        print('dissimilar:', circle_dissimilarity)

        if circle_dissimilarity <= circle_dissimilarity_max:
            shape = "circle"
        else:
            shape = "rectangle"

    return shape, approx_poly

def find_shapes(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = find_blobs(image)

    shape_counts = {
        'circle': 0,
        'triangle': 0,
        'rectangle': 0,
        'square': 0,
    }

    for keypoint in keypoints:
        (lowerX, lowerY) = keypoint['lower']
        (upperX, upperY) = keypoint['upper']

        roi_image = gray[lowerY:upperY, lowerX:upperX]
        threshed_img = cv2.threshold(roi_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find the shapes in the image and sort them from greatest area to least
        shape_cnts = cv2.findContours(threshed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sorted_cnts = sorted(shape_cnts, key=cv2.contourArea, reverse=True)

        if len(sorted_cnts) > 0:
            # There should be only one shape to detect within each ROI image
            shape_cnt = sorted_cnts[0]
            convex_hull = cv2.convexHull(shape_cnt)

            # compute the center of the contour, then detect the name of the shape using only the contour
            area = cv2.contourArea(convex_hull)

            # filter out any shapes that are too big or too small
            if area > max_area or area < min_area:
                continue

            shape, shape_hull = identify_shape(convex_hull)

            if shape == "unidentified":
                continue

            shape_counts[shape] += 1

            cv2.drawContours(
                image,
                [shape_hull + keypoint['center'] - keypoint['size']],
                -1,
                (0, 255, 0),
                2
            )

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