# import the necessary packages
import cv2
import numpy as np

from geometry_utils import unwrap_contour, get_angles_of_shape, get_edge_lengths_of_shape
from blob_detection import find_blobs, draw_found_blobs, imagify_keypoints, combine_images_vertical

min_area = 200
max_area = 20000

circle_max_cnt_length = 6
triangle_length_thresh = 30
square_aspect_ratio_thresh = 2

def identify_shape(cnt):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    perimeter = cv2.arcLength(cnt, True)
    approx_poly = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

    if len(approx_poly) == 2:
        shape = "rectangle"

    # if the shape is a triangle, it will have 3 vertices
    elif len(approx_poly) == 3:
        # we want only roughly equilateral triangles, so we'll calculate the edge lengths,
        # get the biggest difference, and see if it's within a certain threshold
        tri_points = unwrap_contour(approx_poly)

        edge_lengths = get_edge_lengths_of_shape(tri_points)
        max_length_diff = np.max(edge_lengths) - np.min(edge_lengths)
        print('length:', max_length_diff)

        if max_length_diff < triangle_length_thresh:
            shape = "triangle"

    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(approx_poly) == 4:
        rect_points = unwrap_contour(approx_poly)

        # Get nearest/farthest points from origin
        nearest_point = min(rect_points, key=np.linalg.norm)
        farthest_point = max(rect_points, key=np.linalg.norm)

        middle_points = np.array([point for point in rect_points if not np.array_equal(point, nearest_point) and not np.array_equal(point, farthest_point)])

        # Get distances from nearest point to middle 2 points
        middle_near_segments = [(nearest_point, mid_point) for mid_point in middle_points]
        middle_near_distances = [np.linalg.norm(p1 - p2) for p1, p2 in middle_near_segments]
        middle_near_dist_diff = middle_near_distances[0] / middle_near_distances[1]
        if middle_near_dist_diff < 1:
            middle_near_dist_diff = 1 / middle_near_dist_diff

        # Get distances from farthest point to middle 2 points
        middle_far_segments = [(farthest_point, mid_point) for mid_point in middle_points]
        middle_far_distances = [np.linalg.norm(p1 - p2) for p1, p2 in middle_near_segments]
        middle_far_dist_diff = middle_far_distances[0] / middle_far_distances[1]
        if middle_far_dist_diff < 1:
            middle_far_dist_diff = 1 / middle_far_dist_diff

        edges = get_edge_lengths_of_shape(rect_points)
        print('====================')
        print('%.2f %.2f' % (middle_near_dist_diff, middle_far_dist_diff), np.diff(np.sort(edges)))

        shape = "square" if middle_near_dist_diff < square_aspect_ratio_thresh else "rectangle"

    # otherwise, we assume the shape is a circle
    elif len(cnt) >= circle_max_cnt_length:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)

        circle_points = unwrap_contour(cnt)

        angles = get_angles_of_shape(circle_points)
        angles = np.abs(angles - 360 / len(cnt))
        max_angle_diff = np.max(angles) - np.min(angles)
        print(max_angle_diff)

        shape = "circle"

    # return the name of the shape
    return shape, approx_poly

def find_shapes(image, debug=False):
    size = image.shape
    num_pixels = np.prod(image.shape[:2])

    # convert the image to grayscale and threshold it
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
        threshed_img = cv2.adaptiveThreshold(roi_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 39, 15)

        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv2.erode(threshed_img, kernel)

        shape_cnts = cv2.findContours(eroded_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(shape_cnts) > 0:
            shape_cnt = shape_cnts[0]
            convex_hull = cv2.convexHull(shape_cnt)

            # # Remove contours that create invalid polygons
            if len(convex_hull) < 1:
                continue

            # compute the center of the contour, then detect the name of the shape using only the contour
            area = cv2.contourArea(convex_hull)

            # filter out any shapes that are too big or too small
            if area > max_area or area < min_area:
                continue

            shape, shape_hull = identify_shape(convex_hull)

            if shape == "unidentified":
                continue

            shape_counts[shape] += 1

            print('SHAPE FOUND:', shape)

            cv2.drawContours(
                image,
                [convex_hull + keypoint['center'] - keypoint['size']],
                -1,
                (0, 255, 0),
                2
            )

    drawn_image = imagify_keypoints(image, keypoints)

    return drawn_image, list(shape_counts.values())

# Draws the number of found benthic species found on the lower right hand corner
def draw_shape_counter(img, num_circles, num_triangles, num_lines, num_squares):
    text_options = (cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
    # Draw circle counter
    cv2.circle(img, (img.shape[1] - 100, img.shape[0] - 200), 20, (0, 0, 255), -1)
    cv2.putText(img, str(num_circles), (img.shape[1] - 50, img.shape[0] - 180), *text_options)

    # Draw triangle counter
    side_len = 40
    offsetX = img.shape[1] - 120
    offsetY = img.shape[0] - 170

    pt1 = (offsetX + side_len // 2	, offsetY)
    pt2 = (offsetX + 0				, offsetY + side_len)
    pt3 = (offsetX + side_len		, offsetY + side_len)
    triangle_cnt = np.array([pt1, pt2, pt3])

    cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)
    cv2.putText(img, str(num_triangles), (img.shape[1] - 50, img.shape[0] - 130), *text_options)

    # Draw line counter
    cv2.line(img, (img.shape[1] - 120, img.shape[0] - 95), (img.shape[1] - 80, img.shape[0] - 95), (0, 0, 255), 6)
    cv2.putText(img, str(num_lines), (img.shape[1] - 50, img.shape[0] - 80), *text_options)

    # Draw square counter
    (originX, originY) = (img.shape[1] - 120, img.shape[0] - 60)
    side_length = 40

    top_left_coord = (originX, originY)
    bottom_right_coord = (originX + side_length, originY + side_length)

    cv2.rectangle(img, top_left_coord, bottom_right_coord, (0, 0, 255), -1)
    cv2.putText(img, str(num_squares), (img.shape[1] - 50, img.shape[0] - 30), *text_options)

    return img