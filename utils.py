import cv2
import tkinter as tk
root = tk.Tk()
import numpy as np

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
dx = int(screen_width / 3)
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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img_morphology = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_morphology = cv2.dilate(img_morphology, kernel, iterations=1)

    return img_morphology


def define_contour(contours, img_contour):

    for cnt in range(len(contours)):
        perimeter = round(cv2.arcLength(contours[cnt], True))
        if perimeter > 80:
            if perimeter >= 300:
                cv2.drawContours(img_contour, contours[cnt], -1, (0, 128, 0), 2)
            elif 125< perimeter <=130:
                cv2.drawContours(img_contour, contours[cnt], -1, (255, 0, 255), 2)
            elif perimeter < 95:
                cv2.drawContours(img_contour, contours[cnt], -1, (255, 0, 0), 2)
    return img_contour
def detect_false_object(contours, frame, img_contour, threshold):

    img_edge_detection_lap = cv2.Laplacian(frame, cv2.CV_64F)
    abs_edge_lap = np.uint8(np.absolute(img_edge_detection_lap))

    for cnt in range(len(contours)):
        size = len(contours[cnt])
        for j in range(len(contours[cnt])):
            # coordinates of contours
            y = contours[cnt][j][0][0]
            x = contours[cnt][j][0][1]
            if j == 0:
                sum_contour = abs_edge_lap[x, y][0].astype(np.int)
            else:
                sum_contour += abs_edge_lap[x, y][0].astype(np.int)
        mean_contour = round(sum_contour / size)

        perimeter = round(cv2.arcLength(contours[cnt], True))

        if mean_contour < threshold  and 88 < perimeter < 90:
            for j in range(len(contours[cnt])):
                y = contours[cnt][j][0][0]
                x = contours[cnt][j][0][1]
                cv2.circle(img_contour, (y, x), 1, (0, 0, 255), -1)








