from time import time
import mediapipe as mp
import helpers
from helpers import *


def HandTrackingDataCreator():
    TIME_PER_HAND = 20
    start_up = 100

    # if write:
    PromptOverwrite()
    print('Back')
    write = helpers.write
    reset = helpers.reset

    # setWriteResetFalse()  # to not overwrite data on accident
    append = write and not reset
    # Displaying line number upon appending allows for user to know where to delete from data.txt
    # if recording data was unsuccessful
    # Possible additions to code would be to automate clearing data after this line number upon early termination.
    if append:
        with open('data.txt', 'r') as file:
            string = file.read()
            line_number = string.count('\n') + 1
            print('appending from line:', line_number)

    if reset and write:
        with open('data.txt', 'w'):
            pass

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    fps = FPS()
    fingerTimer = FingerTimer()

    with open('data.txt', 'a') as file:
        while fingerTimer.num_fingers <= 5:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            handLandMarks = results.multi_hand_landmarks
            hand_present = handLandMarks

            if hand_present and start_up == 0:
                for handLandMarks in handLandMarks:  # hand
                    centx_list = []
                    centy_list = []
                    id_list = []
                    newline_list = []
                    hand = (id_list, centx_list, centy_list, [fingerTimer.num_fingers] * 21)
                    for id, landmark in enumerate(handLandMarks.landmark):  # point on hand
                        centx, centy = int(landmark.x * w), int(landmark.y * h)

                        centx_list.append(centx)
                        centy_list.append(centy)
                        id_list.append(id)

                        if id % 4 == 0 and id != 0:
                            cv2.circle(img, (centx, centy), 25, (255, 255, 0))
                    mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time

                    np_list = np.array(hand).T
                    if write: np.savetxt(file, np_list, newline='\n')

            else:
                if start_up != 0:
                    fingerTimer.start = time()
                    fingerTimer.cur_time = time()
                    fingerTimer.seconds_passed = 1
                    fingerTimer.time_elapsed = int(fingerTimer.cur_time - fingerTimer.start)
                    # fingerTimer.num_fingers = fingerTimer.seconds_passed // TIME_PER_HAND

            ################################################################################################################
            # DRAW IMAGE
            ################################################################################################################
            h, w, c = img.shape

            if start_up == 0:
                # FPS
                elapsed_time = round(time() - fps.last_update_time, 1)
                fps.currTime = time()
                if elapsed_time >= fps.UPDATE_INTERVAL:
                    fps.last_update_time = time()
                    fps.framesPerSecond = 1 / (fps.currTime - fps.prevTime)
                fps.prevTime = fps.currTime

                ########################################################################################################
                # Put text to image
                ########################################################################################################
                fontSize = 3

                # Show text for indicating whether writing to disk or not
                write_string = 'w:1' if write else 'w:0'
                position = (int(w - 160), int(h - 10))
                cv2.putText(img, str(write_string), position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 3)

                # Finger timer
                timer = int(fingerTimer.cur_time - fingerTimer.start)
                if timer >= 1:
                    fingerTimer.start = time()
                    fingerTimer.seconds_passed += 1
                fingerTimer.cur_time = time()
                fingerTimer.num_fingers = fingerTimer.seconds_passed // TIME_PER_HAND

                # Show text for number of fingers
                position = (int(w * .5), 24 * fontSize)
                cv2.putText(img, str(fingerTimer.num_fingers), position, cv2.FONT_HERSHEY_SIMPLEX, fontSize,
                            (255, 0, 255),
                            3)

                # Show text of timer for each set of fingers
                tph = str(fingerTimer.seconds_passed % TIME_PER_HAND)  # time per hand
                position = (w - 20 * fontSize - (len(tph) - 1) * 20 * fontSize, 24 * fontSize)
                cv2.putText(img, tph, position, cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 3)

                # Show text for FPS
                cv2.putText(img, str(int(fps.framesPerSecond)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
            else:
                # wait to display and begin time calculations. This is done to reduce data imbalance (0 was much less)
                start_up -= 1
                fps.framesPerSecond = 0

            # Draw threshold window for hand positioning.
            # Purpose: model will need unreasonably more data if hand position is not restricted.
            DrawRegion(img)
            cv2.imshow("Image", img)
            k = cv2.waitKey(1)

            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == 27:
                print('ESC')
                cv2.destroyAllWindows()
                break
