import cv2
def updating_background(c_mask, frame, bck, alpha):

    width, height = frame.shape[:2]
    bck_upd = bck.copy()
    for row in range(width):
        for col in range(height):
            v = c_mask[row, col]
            if v == 0:
                p = alpha * frame[row, col] + (1 - alpha) * bck[row, col]
            else:
                p = bck[row, col]
            bck_upd[row, col] = p
    return bck_upd

def trheshold(dif,T):
    width, height = dif.shape[:2]
    tr = dif.copy()
    for row in range(width):
        for col in range(height):
            if dif[row, col] >T:
                po = 255
            else:
                po = 0
            tr[row, col] = po
    return tr

