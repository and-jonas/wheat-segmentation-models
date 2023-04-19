
import numpy as np
import cv2


def get_color_spaces(patch):

    # Scale to 0...1
    img_RGB = np.array(patch / 255, dtype=np.float32)

    # Images are in RGBA mode, but alpha seems to be a constant - remove to convert to simple RGB
    img_RGB = img_RGB[:, :, :3]

    # Convert to other color spaces
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    # weights for TGI
    lambda_r = 670
    lambda_g = 550
    lambda_b = 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # Concatenate all
    descriptors = np.concatenate(
        [img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, np.stack([ExG, ExR, TGI], axis=2)], axis=2)
    # Names
    descriptor_names = ['sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b',
                        'L', 'u', 'v', 'Y', 'U', 'V', 'Y', 'Cb', 'Cr', 'ExG', 'ExR', 'TGI']

    # Return as tuple
    return (img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, ExG, ExR, TGI), descriptors, descriptor_names