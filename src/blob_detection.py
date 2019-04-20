import cv2
import numpy as np

from image_utils import combine_images_vertical

# https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html#simpleblobdetector
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

# Draw a rectangle around all the found BLOBs in an image
def draw_found_blobs(image, keypoints):
    drawn_image = image.copy()

    for i in range(len(keypoints)):
        drawn_image = cv2.rectangle(
            drawn_image,
            keypoints[i]['lower'],
            keypoints[i]['upper'],
            color=(0, 0, 255), thickness=2)

    return drawn_image

# Crop the BLOBs via their lower and upper bounds and append them to the bottom of the input image
def imagify_keypoints(image, keypoints):
    blob_imgs = []
    for keypoint in keypoints:
        (lowerX, lowerY) = keypoint['lower']
        (upperX, upperY) = keypoint['upper']

        # Crop the BLOB, resize it and add to the list of BLOB images
        roi_image = image[lowerY:upperY, lowerX:upperX, :]
        roi_image = cv2.resize(roi_image, (100, 100))
        blob_imgs += [roi_image]

    # Horizontally combine the BLOB images
    blob_stack = np.concatenate((blob_imgs), axis=1)

    # Vertically stack the original image and the BLOB stack
    debug_img = combine_images_vertical([image, blob_stack])
    return debug_img