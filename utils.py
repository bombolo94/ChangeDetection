import cv2
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


