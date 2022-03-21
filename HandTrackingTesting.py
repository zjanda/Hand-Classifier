from time import time
import mediapipe as mp
from helpers import *


def HandTrackingTesting():
    FPS_UPDATE_INTERVAL = 1  # seconds

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils
    fingers = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}

    fps = FPS()

    # load model
    model = load_model('finalized_model.sav')
    votes = []
    string1 = ''
    string2 = ''

    # Finger timer vars
    with open('data.txt', 'a') as file:
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            handLandMarks = results.multi_hand_landmarks
            hand_present = handLandMarks
            h, w, c = img.shape

            if hand_present:
                hand_coords = []
                for handLandMarks in handLandMarks:  # hand
                    for id, landmark in enumerate(handLandMarks.landmark):  # point on hand
                        centx, centy = int(landmark.x * w), int(landmark.y * h)
                        hand_coords.append([id, centx, centy])
                        # if id % 4 == 0 and id != 0:
                        #     cv2.circle(img, (centx, centy), 25, (255, 255, 0))
                    mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)  # draws 1 hand at a time


                hand_coords = np.array(hand_coords) * 1.0
                normalize_hand(hand_coords)
                predictions = model.predict(hand_coords)
                if len(votes) == 10:
                    results = countVotes(votes)
                    vote1 = results["first_vstats"][0]
                    conf1 = results["first_vstats"][1]
                    vote2 = results["second_vstats"][0]
                    conf2 = results["second_vstats"][1]
                    string1 = f'First prediction {vote1}, Second prediction: {vote2}'
                    string2 = f'first pred conf {conf1}, second pred conf: {conf2}'

                    votes = []
                results = countVotes(predictions)['first_vstats'][0]
                # print(vote)
                votes.append(results)

            # FPS
            elapsed_time = round(time() - fps.last_update_time, 1)
            fps.currTime = time()
            if elapsed_time >= FPS_UPDATE_INTERVAL:
                fps.last_update_time = time()
                fps.framesPerSecond = 1 / (fps.currTime - fps.prevTime)
            fps.prevTime = fps.currTime

            # Draw threshold window
            # DrawRegion(img)

            # image, text, pos, font, font size, color, thickness
            cv2.putText(img, str(int(fps.framesPerSecond)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

            # Show Prediction

            cv2.putText(img, string1, (20, h - 25 * 2), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 3)
            cv2.putText(img, string2, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            k = cv2.waitKey(1)

            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == 27:
                print('ESC')
                cv2.destroyAllWindows()
                break
