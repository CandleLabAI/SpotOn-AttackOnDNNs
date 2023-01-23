import numpy as np
from math import log10, sqrt


def extract_salmap(image, thresh_lambda):
    pixels_in_roi = []
    other_pixels = []
    img_max = image.max()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] >= img_max - thresh_lambda:
                pixels_in_roi.append([i, j])
            else:
                other_pixels.append([i, j])

    pixels_in_roi = np.array(pixels_in_roi)
    other_pixels = np.array(other_pixels)

    return pixels_in_roi, other_pixels


def psnr(img, pert_img):
    mse = (img - pert_img) ** 2
    mse = mse.cpu().detach().numpy()
    mse = np.sum(mse)
    if mse == 0:
        return 100
    max_pixel = img.max()
    if max_pixel <= 0:
        max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
