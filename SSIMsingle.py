from skimage.measure import compare_ssim
import imageio
import os
import numpy as np
img1 = imageio.imread("./prediction/p.png")
img2 = imageio.imread("./prediction/phase.png")
img2 = np.resize(img2,(256,256))
img1 = np.resize(img1,(256,256))
ssim = compare_ssim(img1, img2, multichannel=True)
print(ssim)