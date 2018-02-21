import cv2
import tkinter as tk
import numpy as np

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
dx = int(screen_width / 2)
dy = int(screen_height / 2)
label_bar_height = 70

file_output = "output.txt"


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


def morphology(mask):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img_morphology = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_morphology = cv2.dilate(img_morphology, kernel, iterations=1)

    return img_morphology


def define_contour(n_frame, contours, img_contour):

    out_file = open(file_output, "a")
    size = len(contours)

    object_detected = 0

    for cnt in range(size):
        area = round(cv2.contourArea(contours[cnt]))
        perimeter = round(cv2.arcLength(contours[cnt], True))
        classification = " "

        if 540 < area < 590:
            cv2.drawContours(img_contour, contours[cnt], -1, (255, 0, 0), 2)
            classification += "other"
            object_detected += 1
            out_file.write("Object Id: " + str(object_detected) + " | Area: " + str(area) + " | Perimeter: " + str(perimeter) +
                           " | Classification:" + classification + "\n")
        elif 610 < area < 690 or 700 < area <= 730:
            cv2.drawContours(img_contour, contours[cnt], -1, (255, 0, 255), 2)
            classification += "other"
            object_detected += 1
            out_file.write("Object Id: " + str(object_detected) + " | Area: " + str(area) + " | Perimeter: " + str(perimeter) +
                           " | Classification:" + classification + "\n")
        elif 7000 < area < 18000:

            cv2.drawContours(img_contour, contours[cnt], -1, (0, 128, 0), 2)
            classification += "person"
            object_detected += 1
            out_file.write("Object Id: " + str(object_detected) + " | Area: " + str(area) + " | Perimeter: " + str(perimeter) +
                           " | Classification:" + classification + "\n")

    out_file.write("Frame index: " + str(n_frame) + " | Object Detected: " + str(object_detected) + "\n")
    out_file.write("--------------------------------------------------------------\n")

    out_file.close()
    return img_contour


def detect_false_object(contours, gray, background, img_contour):

    sb_x_gray = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3))
    sb_x_background = np.absolute(cv2.Sobel(background, cv2.CV_64F, 1, 0, 3))

    sb_y_gray = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3))
    sb_y_background = np.absolute(cv2.Sobel(background, cv2.CV_64F, 0, 1, 3))

    for cnt in range(len(contours)):
        area = round(cv2.contourArea(contours[cnt]))
        if 500 < area < 740:
            size = len(contours[cnt])
            for j in range(size):
                y = contours[cnt][j][0][0]
                x = contours[cnt][j][0][1]
                if j == 0:

                    sum_sb_x_gray = sb_x_gray[x,y].astype(np.int)
                    sum_sb_x_back = sb_x_background[x, y].astype(np.int)

                    sum_sb_y_gray = sb_y_gray[x, y].astype(np.int)
                    sum_sb_y_back = sb_y_background[x, y].astype(np.int)

                else:

                    sum_sb_x_gray += sb_x_gray[x, y].astype(np.int)
                    sum_sb_x_back += sb_x_background[x, y].astype(np.int)

                    sum_sb_y_gray += sb_y_gray[x, y].astype(np.int)
                    sum_sb_y_back += sb_y_background[x, y].astype(np.int)

            mean_x_gray = round(sum_sb_x_gray/size)
            mean_y_gray = round(sum_sb_y_gray / size)

            mean_x_back = round(sum_sb_x_back / size)
            mean_y_back = round(sum_sb_y_back / size)

            if mean_x_gray < mean_x_back and mean_y_gray < mean_y_back:
                for j in range(len(contours[cnt])):
                    y = contours[cnt][j][0][0]
                    x = contours[cnt][j][0][1]
                    cv2.circle(img_contour, (y, x), 1, (0, 0, 255), -1)










