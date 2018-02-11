import cv2
import tkinter as tk
root = tk.Tk()
import numpy as np

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
dx = int(screen_width / 4)
dy = int(screen_height / 2)
label_bar_height = 70

def show(wait=False, **kwargs):

    x = 0
    y = 0
    for key in kwargs:
        label = key
        img = kwargs[key]
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.imshow(label, img)
        cv2.resizeWindow(label, dx, dy - label_bar_height)
        cv2.moveWindow(label, x, y)
        screen_end = int(x / (screen_width - dx)) > 0
        x = x + dx if not screen_end else 0
        y += dy if screen_end else 0
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def updating_background(c_mask, frame, bck, alpha):
    bck_upd = (alpha*frame + (1-alpha)*bck)*c_mask + bck*(1-c_mask)
    return bck_upd


def denoise(img, n):

    img = cv2.GaussianBlur(img, (n, n), -1)
    img = cv2.medianBlur(img, n)
    return img


def blob_analysis(img_morphology):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = img_morphology.size * (2 / 4)

    params.filterByCircularity = False
    params.minCircularity = 0
    params.maxCircularity = 1

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    rv = 255 - img_morphology
    kp = detector.detect(rv)
    return kp

def morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morphology = cv2.dilate(mask, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morphology = cv2.dilate(img_morphology, kernel, iterations=1)

    return img_morphology

def detect_false_object(cnt, frame,cp):
    lap = cv2.Laplacian(frame, cv2.CV_64F)
    for i in range(len(cnt)):
        y = cnt[i][0][0][0]
        x = cnt[i][0][0][1]
        cv2.circle(lap, (y, x), 3, (0, 255, 0), -1)
        size = len(cnt[i])
        for j in range(len(cnt[i])):
            yj = cnt[i][j][0][0]
            xj = cnt[i][j][0][1]
            #v = lap[xj, yj]
            if j == 0:
                #sG = abs(lap[xj, yj][0])
                sG = abs(lap[xj, yj])
            else:
                #sG += abs(lap[xj, yj][0])
                sG += abs(lap[xj, yj])
        mG = round(sG / len(cnt[i]))
        perimeter = round(cv2.arcLength(cnt[i], True))
        if mG >= 6 and mG<=11 and perimeter>95 and perimeter<110:
            for j in range(len(cnt[i])):
                yj = cnt[i][j][0][0]
                xj = cnt[i][j][0][1]
                cv2.circle(cp, (yj, xj), 2, (0, 255, 0), -1)








