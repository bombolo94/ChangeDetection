import numpy as np
import cv2
import utils as ut

camera = cv2.VideoCapture('video.avi')
nFrame = 0
T = 20
med = 100

ret, frame = camera.read()
if ret is True:
    run = True
else:
    run = False

images_matrices = []

while run:
    ret, frame = camera.read()
    if ret is True:
        cp = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if nFrame < med:
            img = np.asarray(gray)
            images_matrices.append(img)
        else:
            if nFrame == med:
                image_stack = np.concatenate([im[..., None] for im in images_matrices], axis=2)
                background = np.median(image_stack, axis=2)

            foreground = cv2.absdiff(gray.astype(np.uint8), background.astype(np.uint8))
            foreground = ut.denoise(foreground,7)
            c_mask = cv2.threshold(foreground.astype(np.uint8), T, 255, cv2.THRESH_BINARY)[1]

            img_morphology = ut.morphology(c_mask)

            kp = ut.blob_analysis(img_morphology)

            imgBlob = cv2.drawKeypoints(gray, kp, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            rev = img_morphology/255

            rev = 1-rev

            background = ut.updating_background(rev, gray, background, 0.2)

            _, cnt, hierarchy= cv2.findContours(img_morphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(cp, cnt, -1, (0, 0, 255), 1)

            ut.detect_false_object(cnt, gray,cp)

            ut.show(Morpholgy=img_morphology, Contourns=cp)
            cv2.waitKey(100)
            print(nFrame)

        nFrame += 1
    else:
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

'''
DISEGNARE UN PUNTO PER CAPIRE CHE OGGETTO STIAMO CONISDERANDO
y = cnt[i][0][0][0]
#print(y)
x = cnt[i][0][0][1]
#print(x)
f = frame[x,y]
val = lap[x, y]
valx = sobelx[x, y]
valy = sobely[x, y]

#print("-------")
cv2.circle(lap, (y, x), 3, (0, 255, 0), -1)'''
