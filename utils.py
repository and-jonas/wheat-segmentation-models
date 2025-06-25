
import glob
import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

np.seterr(divide='ignore', invalid='ignore')


def list_by_date(data_root, roi_root):

    dates = glob.glob(f'{data_root}/2023*')
    dates = [os.path.basename(x) for x in dates]

    rois = [x for x in glob.glob(f"{roi_root}/*/roi/*.json")]

    JPG = []
    ROI = []
    for d in dates:

        # get all image paths
        jpg_paths = glob.glob(f"{data_root}/{d}/JPEG_cam/*.JPG")
        # get file names
        jpg_files = [os.path.basename(x).replace(".JPG", "") for x in jpg_paths]

        # get corresponding roi paths
        roi_paths = []
        for b in jpg_files:
            a = [x for x in rois if b in x]
            roi_paths.append(a)
        roi_paths = [item for sublist in roi_paths for item in sublist]
        roi_files = [os.path.basename(x).replace(".json", "") for x in roi_paths]

        # ignore image paths, if no corresponding roi path is found
        jpg_idx = [index for index, img_id in enumerate(jpg_files) if img_id in roi_files]
        jpg_paths = [jpg_paths[i] for i in jpg_idx]

        JPG.append(jpg_paths)
        ROI.append(roi_paths)

    return JPG, ROI

def center_crop(img_array: np.ndarray, crop_size: int = 4000) -> np.ndarray:
    h, w = img_array.shape[:2]

    if h < crop_size or w < crop_size:
        raise ValueError(f"Image is too small ({w}x{h}) for a {crop_size}x{crop_size} center crop.")

    x_start = (w - crop_size) // 2
    y_start = (h - crop_size) // 2

    return img_array[y_start:y_start + crop_size, x_start:x_start + crop_size]


def crop_images(date_images, date_rois):

    # check that there is a json for each jpeg
    if len(date_images) != len(date_rois):
        print("series not of equal length!")
        print(date_images)
        print(date_rois)

    leaf_crops = []
    image_ids = []
    # for j in range(len(date_images)):
    for j in range(1):

        print(j)

        # get image id
        # image_id = os.path.basename(date_images[j]).replace(".JPG", "")
        image_path = date_images[j]

        image_id = os.path.basename(image_path)

        # get img
        img = Image.open(image_path)
        img = np.asarray(img)

        # get roi coordinates and rotation
        f = open(date_rois[j])
        data = json.load(f)
        rot = np.asarray(data['rotation_matrix'])
        bbox = np.asarray(data['bounding_box'])
        f.close()

        # rotate the image
        rows, cols = img.shape[0], img.shape[1]
        img_rot = cv2.warpAffine(img, rot, (cols, rows))

        # get the leaf
        box_ = np.intp(bbox)
        leaf = img_rot[box_[0][1] - 50:box_[2][1] + 50, 0:8192]

        height = leaf.shape[0]
        margin = height % 32
        leaf = cv2.copyMakeBorder(leaf, margin, 0, 0, 0, cv2.BORDER_CONSTANT)

        # write cropped leaf image
        crop_path = Path(*Path(image_path).parts[:-2]) / "leaf" / image_id
        img = Image.fromarray(leaf)
        img.save(crop_path)

        leaf_crops.append(leaf)
        image_ids.append(image_path)

    return leaf_crops, image_ids


def preprocess(crops):

    pp_crops = []
    for c in crops:
        # pad
        height = c.shape[0]
        margin = 32-(height % 32)
        c = cv2.copyMakeBorder(c, margin, 0, 0, 0, cv2.BORDER_CONSTANT)
        # convert
        c = np.moveaxis(c, 2, 0) / 255.0
        pp_crops.append(c)

    return pp_crops


# vegetation index
def calculate_index(img):
    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img)

    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))

    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    return ExG


def color_index_transformation(image):
    R, G, B = cv2.split(image)
    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))
    normalizer[normalizer == 0] = np.float32(1)  # avoid division by 0
    r, g, b = (R, G, B) / normalizer
    # Green Leaf Index
    GLI = np.array((2 * g - r - b) / (2 * g + r + b), dtype=np.float32)
    # Excess Green Index
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)
    # Excess Red Index
    ExR = np.array(1.3 * r - g)
    # Normalized Difference Index
    NDI = 128 * (((g - r) / (g + r)) + 1)
    # Excess Green minus Excess red Index
    ExGR = (2 * g - (r + b)) - (1.3 * r - g)
    # Normalized Green Red Difference Index
    NGRDI = np.array((g - r) / (g + r), dtype=np.float32)
    # Triangular greenness index
    lambda_R = 670
    lambda_G = 550
    lambda_B = 480
    TGI = -0.5 * ((lambda_R - lambda_B) * (r - g) - (lambda_R - lambda_G) * (r - b))
    # Vegetation Index
    r[r == 0] = 0.00001  # Avoid division by zero
    b[b == 0] = 0.00001  # Avoid division by zero
    VEG = g / ((r ** 0.667) * (b ** 0.333))

    desc = [GLI, ExG, ExR, NDI, ExGR, NGRDI, TGI, VEG]
    desc_names = ['GLI', 'ExG', 'ExR', 'NDI', 'ExGR', 'NGRDI', 'TGI', 'VEG']

    return desc, desc_names


def index_distribution(image, feature_name, level_id, level_mask):
    px_roi = image[level_mask == 255]
    mn = np.nanmean(px_roi)
    md = np.nanmedian(px_roi)
    kt = kurtosis(px_roi, nan_policy='omit')
    sk = skew(px_roi, nan_policy='omit')
    if not px_roi.size == 0:
        p75, p25 = np.nanpercentile(px_roi, [75, 25])
        iqr = p75 - p25
        p98, p02 = np.nanpercentile(px_roi, [98, 2])
        ipr = p98 - p02
    else:
        iqr = np.nan
        ipr = np.nan
    std = np.nanstd(px_roi)
    stat_names = ["mean", "median", "kurtosis", "skewness", "intqrange", "intprange", "stddev"]
    stat_names = [level_id + "_" + feature_name + "_" + n for n in stat_names]
    return [mn, md, kt, sk, iqr, ipr, std], stat_names


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size - overlap)
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


