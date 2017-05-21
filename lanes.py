#!/usr/bin/env python3

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('camera_undistort_matrix.pkl', 'rb') as f:
    (camera_undistort_matrix, camera_undistort_dist) = pickle.load(f)

def loadUndistortedImageAsYUV(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2YUV)
    img = cv2.undistort(img, camera_undistort_matrix, camera_undistort_dist, None, camera_undistort_matrix)
    return img

def showImage(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_YUV2RGB)
    plt.imshow(img)
    plt.show()


