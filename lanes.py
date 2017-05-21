#!/usr/bin/env python3

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


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

def drawLines(img, points):
    pts = points.reshape((-1,1,2))
    img = cv2.polylines(img, [pts], True, (0,255,255),thickness=2)
    return img


def warper(img, debugDrawLines=False, debugDoNotTransform=False):
    scale = np.divide(np.array([1280, 720]), [img.shape[1],img.shape[0]])
    src_points = np.float32([[253,678],[592,450],[687,450],[1054,678]] * scale)
    dst_points = np.float32([[320,700],[320,0],[960,0],[960,700]] * scale)
    if debugDrawLines and debugDoNotTransform:
        return drawLines(img, np.int32(src_points))
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    img = cv2.warpPerspective(img,M,(1280,720))
    if debugDrawLines:
        img = drawLines(img, np.int32(dst_points))
    return img

def warperDebug(img, filename):
    imgBefore = warper(np.copy(img), debugDrawLines=True, debugDoNotTransform=True)
    imgAfter = warper(np.copy(img), debugDrawLines=True, debugDoNotTransform=False)
    img = np.concatenate((imgBefore,imgAfter),axis=1)
    saveImage(img, filename)

def laneCurve(coeff, y):
    return coeff[0] * y*y + coeff[1]*y +coeff[2]

def fitPolynomialToLane(img, x0):
    img_height = img.shape[0]
    img_width = img.shape[1]
    def residuals(coeff):
        r = []
        for y in range(img_height):
            target_x = laneCurve(coeff, y)
            for x in range(int(img_width/2)):
                v = img[y,x,2]
                if v > 0:
                    dist = x - target_x
                    r.append(dist)
        return np.array(r)

    res_log = optimize.least_squares(residuals, x0, loss='cauchy', f_scale=0.1)
    print(res_log)
    return res_log.x

def drawLane(img,X):
    print(img.shape)
    for y in range(img.shape[0]):
        x = int(laneCurve(X, y))
        img[y,x:(x+4)] = [40,255,255]
    return img

img = loadUndistortedImageAsHSV('test_images/straight_lines1.jpg')

#warperDebug(img, "warper1.png")
img = sobelx(img)
#saveImage(img, "sobelx.png")
img = applyThresholds(img)
#saveImage(img, "thresholded.png")
#warperDebug(img, "warper2.png")

img = loadUndistortedImageAsHSV('test_images/test2.jpg')
img = sobelx(img)
img = applyThresholds(img)
img = warper(img)
X = np.array([0,0,201])
X = fitPolynomialToLane(img, X)
img = drawLane(img, X)
showImage(img)