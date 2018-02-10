import numpy as np
import cv2
import utils as ut

camera = cv2.VideoCapture('video.avi')
nFrame = 0
T = 40
med = 100

ret, frame = camera.read()
if ret is True:
    run = True
else:
    run = False

while run:
    ret, frame = camera.read()
    if ret is True:
        cp = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if nFrame < med:
            if nFrame == 0:
                prev = gray
            s = cv2.add(gray.astype(np.int32), prev.astype(np.int32))
            prev = s

        else:
            if nFrame == med:
                backgroundI = (s / med)
                background = backgroundI

            foreground = cv2.absdiff(gray.astype(np.uint8), background.astype(np.uint8))
            foreground = cv2.GaussianBlur(foreground, (7,7),-1)
            foreground = cv2.medianBlur(foreground, 7)
            imgMorphology = cv2.threshold(foreground.astype(np.uint8), T, 255, cv2.THRESH_BINARY)[1]

            # imgMorphology = cv2.medianBlur(imgMorphology, 17)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            imgMorphology = cv2.dilate(imgMorphology, kernel, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_CLOSE, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            imgMorphology = cv2.dilate(imgMorphology, kernel, iterations=1)

            # Blob Analysis

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 0
            params.maxThreshold = 256

            params.filterByArea = True
            params.minArea = 100
            params.maxArea = imgMorphology.size * (2 / 4)

            params.filterByCircularity = False
            params.minCircularity = 0
            params.maxCircularity = 1

            params.filterByInertia = False
            params.filterByConvexity = False
            params.filterByColor = False

            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs
            rv = 255 - imgMorphology
            kp = detector.detect(rv)

            imgBlob = cv2.drawKeypoints(gray, kp, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # End Blob Analysis

            # FindCount
            background = ut.updating_background(imgMorphology, gray, background, 0.03)

            _, cnt, hierarchy= cv2.findContours(imgMorphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


            lap = cv2.Laplacian(frame, cv2.CV_64F)
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            for i in range(len(cnt)):
                #if nFrame > 498:
                    # cnt[0][0][0][0] è 256 e cnt[0][0][0][1] è 40
                    size = len(cnt[i])
                    for j in range(len(cnt[i])):
                        yj = cnt[i][j][0][0]
                        xj = cnt[i][j][0][1]
                        v = lap[xj, yj][0]
                        if j == 0:
                            sG = lap[xj, yj][0]
                        else:
                            sG += lap[xj, yj][0]
                    mG = round(abs(sG/len(cnt[i])))
                    perimeter = round(cv2.arcLength(cnt[i], True))
                    if mG <=1:
                        if perimeter > 90:
                            if perimeter < 115:
                                for j in range(len(cnt[i])):
                                    yj = cnt[i][j][0][0]
                                    xj = cnt[i][j][0][1]
                                    cv2.circle(frame, (yj, xj), 2, (0, 255, 0), -1)
                                #ut.show(Laplacian=lap)
                                #cv2.waitKey(0)
                    #ut.show(Laplacian=lap)
                    #cv2.waitKey(0)
                    cv2.drawContours(cp, cnt[i], -1, (0, 0, 255), 1)
            # color = np.random.randint(0, 255, (3)).tolist()

            ut.show(Morpholgy=imgMorphology, Contourns=frame, Blob= imgBlob)
            cv2.waitKey(10)
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