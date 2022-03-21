import pickle
import tkinter.messagebox
from tkinter import *
from tkinter.ttk import *
import cv2
import os
import numpy as np
from time import time

from icecream import ic

write = True
reset = False


# FPS vars
class FPS:
    UPDATE_FREQ = 1  # seconds
    prevTime = 0
    currTime = time()
    last_update_time = time()
    framesPerSecond = 0


# Finger timer vars
class FingerTimer:
    TIME_PER_HAND = 20
    start = time()
    cur_time = time()
    seconds_passed = 1
    time_elapsed = int(cur_time - start)
    num_fingers = seconds_passed // TIME_PER_HAND



def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


def DrawRegion(img):
    h, w, c = img.shape
    top = .27
    bot = 1 - top
    top_x = int(top * w)
    top_y = int(top * h - 150)
    bot_x = int(bot * w)
    bot_y = int(bot * h)
    x_os = 150  # offset
    y_os = 70  # offset
    TH_TOPLEFT = (top_x + x_os, top_y + y_os)
    TH_BOTRIGHT = (bot_x + x_os, bot_y + y_os)
    cv2.rectangle(img, TH_TOPLEFT, TH_BOTRIGHT, (0, 0, 0), 10)


def countVotes(predictions):
    hist = np.histogram(predictions, bins=6, range=[0, 5])[0]

    conf1 = np.max(hist) / np.sum(hist)
    first = np.argmax(hist)
    first_ct = hist[first]

    hist[first] = 0
    second = np.argmax(hist)
    conf2 = np.max(hist) / (np.sum(hist) + first_ct)
    return {'first_vstats': (first, conf1), 'second_vstats': (second, conf2)}


def PromptOverwrite():
    with open('data.txt', 'r') as file:
        if file.read():
            master = Tk()
            master.eval('tk::PlaceWindow . center')
            master.geometry("200x150")
            style = Style()
            style.configure('W.TButton', font=('calibri', 10, 'bold'))
            Label(master, text="Write to disk?").pack(pady=10)

            def updateWrite(value):
                global write
                write = value
                master.destroy()

                if write:
                    reset_window = Tk()
                    Label(reset_window, text='Reset Data?').pack(pady=10)

                    def updateReset(value):
                        global reset
                        reset = value
                        reset_window.destroy()

                    Button(reset_window, style='W.TButton', text="Yes", command=lambda *args: updateReset(True)).pack(pady=10)
                    Button(reset_window, style='W.TButton', text="No", command=lambda *args: updateReset(False)).pack(pady=10)

                    mainloop()
                else:
                    print('return')
                    return
            Button(master, style='W.TButton', text="Yes", command=lambda *args: updateWrite(True)).pack(pady=10)
            Button(master, style='W.TButton', text="No", command=lambda *args: updateWrite(False)).pack(pady=10)
            print('return 2')
            def on_closing():
                # if tkinter.messagebox.askokcancel('Quit', 'Do you want to quit?'):
                master.destroy()
                exit(0)


            master.protocol('WM_DELETE_WINDOW', on_closing)

            # mainloop, runs infinitely
            mainloop()


def setWriteResetFalse():
    with open('HandTrackingDataCreator.py', 'r') as file:
        string = file.read()

    string = string.replace('write = True', 'write = False', 1)
    string = string.replace('reset = True', 'reset = False', 1)

    with open('HandTrackingDataCreator.py', 'w') as file:
        file.write(string)

# Detect if a point is inside the threshold
# def in_threshold(centx, centy):
#     return TH_TOPLEFT[0] < centx < TH_BOTRIGHT[0] and TH_TOPLEFT[1] < centy < TH_BOTRIGHT[1]


def normalize_hand(hand: type(np.array([]))):
    ic(hand.shape)
    x = hand[:, 1] - np.min(hand[:, 1])
    hand[:, 1] = x / np.max(x)
    y = hand[:, 2] - np.min(hand[:, 2])
    hand[:, 2] = y / np.max(y)
    return hand
