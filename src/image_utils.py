import numpy as np

# Stack a series of images of varying widths vertically (for debugging purposes)
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
    for img in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:current_y + img.shape[0], :img.shape[1], :] = image
        current_y += img.shape[0]

    return final_image