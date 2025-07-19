import cv2
import numpy as np
import copy
import math
import random

def printThreshold(thr):
    print("! Changed threshold to " + str(thr))

def Game():
    bgModel = None
    cap_region_x_begin = 0.5
    cap_region_y_end = 0.8
    threshold = 60
    blurValue = 41
    bgSubThreshold = 50
    learningRate = 0

    isBgCaptured = 0
    triggerSwitch = False

    print("press 'b' to capture your background.")
    print("press 'n' to capture your gesture.")

    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 200, printThreshold)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        if isBgCaptured == 1:
            fgmask = bgModel.apply(frame, learningRate=learningRate)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            img = cv2.bitwise_and(frame, frame, mask=fgmask)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                      int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            cv2.imshow('mask', img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            cv2.imshow('blur', blur)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            cv2.imshow('ori', thresh)

            thresh1 = copy.deepcopy(thresh)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maxArea = -1
            ci = -1

            for i in range(len(contours)):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea and area >= 1000:
                    maxArea = area
                    ci = i

            if ci != -1:
                res = contours[ci]
                drawing = np.zeros(img.shape, np.uint8)
                if len(res) >= 3:
                    hull = cv2.convexHull(res)
                    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                hull_indices = cv2.convexHull(res, returnPoints=False) if len(res) >= 5 else None
                Flag = True
                if hull_indices is not None and len(hull_indices) > 3:
                    try:
                        defects = cv2.convexityDefects(res, hull_indices)
                    except cv2.error as e:
                        print("convexityDefects failed:", e)
                        defects = None

                    if defects is not None:
                        cnt = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i][0]
                            start = tuple(res[s][0])
                            end = tuple(res[e][0])
                            far = tuple(res[f][0])
                            a = math.dist(start, end)
                            b = math.dist(start, far)
                            c = math.dist(end, far)
                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                            if angle <= math.pi / 2 and d > 10000:
                                cnt += 1
                                cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                        isFinishCal, cnt, Flag = True, cnt, False
                if Flag != False:
                    isFinishCal, cnt = False, 0

                if triggerSwitch is True:
                    if isFinishCal is True and cnt <= 5:
                        if cnt == 0:
                            print("stone")
                            gesture = 0
                        elif cnt <= 2:
                            print("scissors")
                            gesture = 1
                        else:
                            print("paper")
                            gesture = 2
                        camera.release()
                        cv2.destroyAllWindows()
                        break

                cv2.imshow('output', drawing)

        k = cv2.waitKey(10)
        if k == 27:
            camera.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=bgSubThreshold, detectShadows=False)
            isBgCaptured = 1
            print('!!!Background Captured!!!')
        elif k == ord('r'):
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print('!!!Reset BackGround!!!')
        elif k == ord('n'):
            triggerSwitch = True
            print('!!!Trigger On!!!')

    play = ['rock', 'scissors', 'paper']
    p1 = gesture
    pc = random.randint(0, 2)
    print(f"you are {play[p1]}, and the computer is {play[pc]}")
    if p1 == pc:
        print("Game Draw")
        game_result = 1
    elif (p1 == 0 and pc == 1) or (p1 == 1 and pc == 2) or (p1 == 2 and pc == 0):
        print("you win!")
        game_result = 1
    else:
        print("you lose!")
        game_result = -1
    return game_result

Game()