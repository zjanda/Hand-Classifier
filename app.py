from helpers import *
from tkinter.ttk import *
from multiprocessing import Process
from HandTrackingTesting import HandTrackingTesting
from HandTrackingDataCreator import HandTrackingDataCreator


#!/bin/python
# os.system('python3 HandTrackingDataCreator.py')
def runRecordData():
    rec_data_window = Tk()
    num_list = []
    def selectNumFingers(value):
        global num_list
        num_list.append(value)
    Label(rec_data_window, text='Select fingers to record')
    Checkbutton(rec_data_window, text='0', command=lambda *argv: selectNumFingers(0))
    Checkbutton(rec_data_window, text='1', command=lambda *argv: selectNumFingers(1))
    Checkbutton(rec_data_window, text='2', command=lambda *argv: selectNumFingers(2))
    Checkbutton(rec_data_window, text='3', command=lambda *argv: selectNumFingers(3))
    Checkbutton(rec_data_window, text='4', command=lambda *argv: selectNumFingers(4))
    Checkbutton(rec_data_window, text='5', command=lambda *argv: selectNumFingers(5))
    p1 = Process(target=lambda *argv: HandTrackingDataCreator(num_list))
    p1.start()
    p1.join()


def runTrainModel():
    print(os.system('jupyter nbconvert --to script modeltrain.ipynb'))
    import modeltrain


def runTestModel():
    p1 = Process(target=HandTrackingTesting)
    p1.start()
    p1.join()


if __name__ == '__main__':
    mainWindow = Tk()

    app_name = 'Zack\'s Fantastic Hand Classifier'
    mainWindow.title(app_name)
    # mainWindow.geometry('350x220')
    mainWindow.eval('tk::PlaceWindow . center')
    style = Style()
    style.configure('W.TButton', font=('calibri', 10, 'bold'))
    Label(mainWindow, text=f'Welcome to {app_name}').pack(pady=20)
    Button(mainWindow, text="Record Data", style='W.TButton', command=runRecordData).pack(pady=10, padx=130)
    Button(mainWindow, text="Train Model", style='W.TButton', command=runTrainModel).pack(pady=10)
    Button(mainWindow, text="Test Model", style='W.TButton', command=runTestModel).pack(pady=10)

    mainloop()
