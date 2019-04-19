import cv2
import numpy as np

# https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html#simpleblobdetector
def find_blobs(image):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.minRepeatability = 4 # higher number yields stabler blobs

    max_min_repeatability = int((params.maxThreshold - params.minThreshold) / params.thresholdStep)
    if params.minRepeatability >= max_min_repeatability:
        raise Exception('Max "minRepeatability" (%s) is exceeded by given minRepeatability (%s)' % (max_min_repeatability, params.minRepeatability))

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 18000

    params.filterByCircularity = False
    params.filterByInertia = False

    params.filterByConvexity = True
    params.minConvexity = 0.9

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    keypoints = clean_keypoints(keypoints)
    keypoints = keypoint_rect_bounds(keypoints, image.shape)

    return keypoints

def clean_keypoints(keypoints):
    cleaned_keypoints = []
    for keypoint in keypoints:
        cleaned_keypoints += [{
            'center': (np.int(keypoint.pt[0]), np.int(keypoint.pt[1])),
            'size': np.int(keypoint.size / 1.1),
            'lower': None,
            'upper': None
        }]

    return cleaned_keypoints

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

def keypoint_rect_bounds(keypoints, img_shape):
    for i in range(len(keypoints)):
        lower, upper = get_keypoint_bounds(keypoints[i], img_shape)
        keypoints[i]['lower'] = lower
        keypoints[i]['upper'] = upper

    return keypoints

def draw_found_blobs(image, keypoints):
    drawn_image = image.copy()

    for i in range(len(keypoints)):
        drawn_image = cv2.rectangle(
            drawn_image,
            keypoints[i]['lower'],
            keypoints[i]['upper'],
            color=(0, 0, 255), thickness=2)

    return drawn_image

def imagify_keypoints(image, keypoints):
    if len(keypoints) <= 0:
        return image

    blob_imgs = []
    for keypoint in keypoints:
        (lowerX, lowerY) = keypoint['lower']
        (upperX, upperY) = keypoint['upper']

        roi_image = image[lowerY:upperY, lowerX:upperX, :]
        roi_image = cv2.resize(roi_image, (100, 100))
        blob_imgs += [roi_image]

    blob_stack = np.concatenate((blob_imgs), axis=1) # horizontally combine the images
    return combine_images_vertical([image, blob_stack])

def combine_images_vertical(images):
    widths = []
    max_height = 0

    for img in images:
        widths.append(img.shape[1])
        max_height += img.shape[0]

    w = np.max(widths)
    h = max_height

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((h, w, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:current_y + image.shape[0], :image.shape[1], :] = image
        current_y += image.shape[0]

    return final_image