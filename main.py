import numpy as np
import cv2
import utils as ut

camera = cv2.VideoCapture('video.avi')
nFrame = 0
threshold = 20
value = 100

ret, frame = camera.read()
if ret is True:
    run = True
else:
    run = False

images_matrices = []

while run:
    ret, frame = camera.read()
    if ret is True:

        img_contour = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if nFrame < value:
            img = np.asarray(gray)
            images_matrices.append(img)
        else:
            if nFrame == value:
                image_stack = np.concatenate([im[..., None] for im in images_matrices], axis=2)
                background = np.median(image_stack, axis=2)

            foreground = cv2.absdiff(gray.astype(np.uint8), background.astype(np.uint8))
            foreground = ut.denoise(foreground,7)
            c_mask = cv2.threshold(foreground.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)[1]

            img_morphology = ut.morphology(c_mask)

            kp = ut.blob_analysis(img_morphology)

            imgBlob = cv2.drawKeypoints(gray, kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            rev = img_morphology/255

            rev = 1-rev

            background = ut.updating_background(rev, gray, background, 0.1)

            _, cnt, hierarchy= cv2.findContours(img_morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 1)
            if nFrame>=268:
                img_contour = ut.define_contour(cnt, img_contour)

            ut.detect_false_object(cnt, frame, img_contour, threshold)
            ut.show(Morpholgy=img_morphology, Contours=img_contour, Frame= frame)
            cv2.waitKey(0)
            print(nFrame)

        nFrame += 1
    else:
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

