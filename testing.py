
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

img1 = Image.open("Z:/Public/Jonas/Data/EXT/5/FPWW0340202_FIP2_cam_04.png")
img1 = np.asarray(img1)

img0 = Image.open("Z:/Public/Jonas/Data/ESWW010/ImagesNadir/20240717/JPEG_cam/20240717_095421_Cam_ESWW00100038_Cnp_1.JPG")
img0 = np.asarray(img0)

plt.imshow(img0)