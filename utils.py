import cv2
import tkinter as tk
root = tk.Tk()
import numpy as np

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
dx = int(screen_width / 4)
dy = int(screen_height / 2)
label_bar_height = 65


def subst(img, T):

    imgo = img.copy()
    width, height = img.shape[:2]
    for row in range(width):
        for col in range(height):
            if img[row, col] > T:
                imgo[row, col] = 255
            else:
                img[row, col] = 0
    return imgo


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

    width, height = frame.shape[:2]
    bck_upd = bck.copy()
    for row in range(width):
        for col in range(height):
            if c_mask[row, col] == 0:
                p = alpha * frame[row, col] + (1 - alpha) * bck[row, col]
            else:
                p = bck[row, col]
            bck_upd[row, col] = p
    return bck_upd


