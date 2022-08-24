import os

from HandTrackingDataCreator import HandTrackingDataCreator
from HandTrackingTesting import HandTrackingTesting
from helpers import *


# !/bin/python
# os.system('python3 HandTrackingDataCreator.py')
def runRecordData():
    # rec_data_window = Tk()
    num_list = []
    #
    # Label(rec_data_window, text='Select fingers to record')
    # Checkbutton(rec_data_window, text='0', command=lambda *argv: num_list.append(0))
    # Checkbutton(rec_data_window, text='1', command=lambda *argv: num_list.append(1))
    # Checkbutton(rec_data_window, text='2', command=lambda *argv: num_list.append(2))
    # Checkbutton(rec_data_window, text='3', command=lambda *argv: num_list.append(3))
    # Checkbutton(rec_data_window, text='4', command=lambda *argv: num_list.append(4))
    # Checkbutton(rec_data_window, text='5', command=lambda *argv: num_list.append(5))
    HandTrackingDataCreator(num_list)


def runTrainModel():
    try:
        import DNNmodeltrain
    except ModuleNotFoundError:
        print('No model trained...')
        print('Generating model training instructions...')
        print(os.system('jupyter nbconvert --to script DNNmodeltrain.ipynb'))
        import DNNmodeltrain
        # print('File created. Please try again.')


def runTestModel():
    HandTrackingTesting()


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
