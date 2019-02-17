#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Philipp
#
# Created:     16.02.2019
# Copyright:   (c) Philipp 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import numpy as np

def grapeDetection():
    img = cv2.imread(r".\data\Auswahl\P63_R2_r___Data_2017_09_13_084114_ERO\063_002_036_1DC24E74_1505292356747.jpg")
    thresh = 40
    thresh_max = 80
    maxval = 1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img_gray, s, v = cv2.split(img_gray)
    #img_gray = cv2.resize(img_gray, (0, 0), fx = 0.4, fy = 0.4)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    print(img_gray[0,200])
    cv2.imshow("Results", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    _, t_img = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY)
    _, t_img2 = cv2.threshold(img_gray, thresh_max, maxval, cv2.THRESH_TOZERO_INV)

    t_img = img_gray
    #t_img = t_img+t_img2-127

    c_vec = cv2.HoughCircles(t_img, method = cv2.HOUGH_GRADIENT, dp = 8, minDist = 1, maxRadius = 30)
    #c_vec2 = cv2.HoughCircles(t_img, method = cv2.HOUGH_GRADIENT, dp = 4, minDist = 15, maxRadius = 30)
    #c_vec3 = cv2.HoughCircles(t_img, method = cv2.HOUGH_GRADIENT, dp = 4.5, minDist = 15, maxRadius = 30)

    r_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

    try:
        for circle in c_vec[0]:
            mask = np.full((t_img.shape[0], t_img.shape[1]), 0, dtype = np.uint8)
            cir = cv2.circle(mask, (circle[0], circle[1]), circle[2], (255,255,255), -1)
            if cv2.mean(t_img, mask)[0] < 0:
               np.delete(c_vec[0], circle)
               continue
            cir = cv2.circle(r_img, (circle[0], circle[1]), circle[2], (255,0,255))
            r_img = cir

    except:
        print("Error!")

    try:
        for circle in c_vec2[0]:
            mask = np.full((t_img.shape[0], t_img.shape[1]), 0, dtype = np.uint8)
            cir = cv2.circle(mask, (circle[0], circle[1]), circle[2], (255,255,255), -1)
            if cv2.mean(t_img, mask)[0] < 0:
               np.delete(c_vec[0], circle)
               continue
            cir = cv2.circle(r_img, (circle[0], circle[1]), circle[2], (255,0,255))
            r_img = cir

    except:
        print("Error!")

    try:
        for circle in c_vec3[0]:
            mask = np.full((t_img.shape[0], t_img.shape[1]), 0, dtype = np.uint8)
            cir = cv2.circle(mask, (circle[0], circle[1]), circle[2], (255,255,255), -1)
            if cv2.mean(t_img, mask)[0] < 50:
               np.delete(c_vec[0], circle)
               continue
            cir = cv2.circle(r_img, (circle[0], circle[1]), circle[2], (255,0,255))
            r_img = cir

    except:
        print("Error!")

    r_img = cv2.resize(r_img, (0, 0), fx = 0.4, fy = 0.4)
    cv2.imshow("Results", r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    grapeDetection()

if __name__ == '__main__':
    main()
