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
            foreground = cv2.GaussianBlur(foreground, (15,15),-1)
            c_mask = cv2.threshold(foreground.astype(np.uint8), T, 255, cv2.THRESH_BINARY)[1]

            # imgMorphology = cv2.medianBlur(imgMorphology, 17)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            imgMorphology = cv2.dilate(c_mask, kernel, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
            imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_CLOSE, kernel)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # imgMorphology = cv2.morphologyEx(imgMorphology, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            imgMorphology = cv2.dilate(imgMorphology, kernel, iterations=0)

            # Blob Analysis

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 0
            params.maxThreshold = 256

            params.filterByArea = True
            params.minArea = 50
            params.maxArea = imgMorphology.size * (1 / 4)

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

            # kp is a list of kp, which include the coordinates (of the centres) of the blobs,
            #  and their size

            imgBlob = cv2.drawKeypoints(gray, kp, np.array([]), (0, 0, 155), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # End Blob Analysis

            # FindCount
            background = ut.updating_background(imgMorphology, gray, background, 0.03)

            _, cnt, hierarchy= cv2.findContours(imgMorphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


            lap = cv2.Laplacian(frame, cv2.CV_64F)
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            for i in range(len(cnt)):
                if nFrame == 123:
                    y = cnt[i][0][0][0]
                    x = cnt[i][0][0][1]
                    f = frame[x, y]
                    val = lap[x, y]
                    valx = sobelx[x, y]
                    valy = sobely[x, y]
                    print(f)
                    print(val)
                    print(valx)
                    print(valy)
                    print("-------")
                    cv2.circle(lap, (y, x), 6, (0, 0, 255), -1)
                    ut.show(Laplacian=lap)
                    cv2.waitKey(0)
                if nFrame == 335:
                    # cv[0][0][0] è 256 e cv[0][0][1] è 40
                    y = cnt[i][0][0][0]
                    #print(y)
                    x = cnt[i][0][0][1]
                    #print(x)
                    f = frame[x,y]
                    val = lap[x, y]
                    valx = sobelx[x, y]
                    valy = sobely[x, y]
                    print(f)
                    print(val)
                    print(valx)
                    print(valy)
                    print("-------")
                    cv2.circle(lap, (y, x), 6, (0, 0, 255), -1)
                    ut.show(Laplacian=lap)
                    cv2.waitKey(0)
                cv2.drawContours(cp, cnt[i], -1, (0, 0, 255), 2)
            # color = np.random.randint(0, 255, (3)).tolist()



            ut.show(Laplacian=lap ,Morpholgy=imgMorphology, Contourns=frame)
            cv2.waitKey(1)
            print(nFrame)

        nFrame += 1
    else:
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

''' 
            if nFrame == 334:
                
                for i in range(len(cnt)):
                    cn = cnt[i]
                    x, y, w, h = cv2.boundingRect(cn)
                    cv2.rectangle(cp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    val = lap[y, x].astype(np.uint8)
                    f = frame[y,x].astype(np.uint8)
                    print(val)
                    print("- ")
                    print(f)
                    print("---------")
                    rect = cv2.minAreaRect(cn)
                    box = cv2.boxPoints(rect)
                    # convert all coordinates floating point values to int
                    box = np.int0(box)
                    cv2.drawContours(cp, [box], 0, (0, 0, 255))
                    ut.show(Prova=cp)
                    cv2.waitKey(0)'''


