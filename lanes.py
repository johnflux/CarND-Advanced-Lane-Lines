#!/usr/bin/env python3

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('camera_undistort_matrix.pkl', 'rb') as f:
    (camera_undistort_matrix, camera_undistort_dist) = pickle.load(f)

def loadUndistortedImageAsHSV(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2HSV)
    img = cv2.undistort(img, camera_undistort_matrix, camera_undistort_dist, None, camera_undistort_matrix)
    return img

def saveImage(img, filename):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_HSV2BGR)
    cv2.imwrite("output_images/" + filename, img)

def showImage(img):
    if img.ndim == 3:
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_HSV2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()

def sobelx(img):
    sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0)
    abs_sobelx = np.abs(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    img[:, :, 2] = scaled_sobelx
    return img

def applyThresholds(img):
    H = img[:,:,0]
    S = img[:,:,1]
    V = img[:,:,2]
    mask = np.logical_or(np.logical_and(np.logical_or(H < 15, H > 30), S > 20), V < 50)
    img[mask,2] = 0
    return img

def warper(img):
    scale = np.divide(np.array([720, 1280]), img.shape[0:2])
    src_points = np.float32([[264,678],[626,431],[1038,678],[651,431]]) * scale
    dst_points = np.float32([[320,0],[320,720],[960,720],[960,0]]) * scale
    print(src_points)

    return img

img = loadUndistortedImageAsHSV('test_images/straight_lines1.jpg')
img = sobelx(img)

saveImage(img, "sobelx.png")
img = applyThresholds(img)
saveImage(img, "thresholded.png")

img = warper(img)
showImage(img)

