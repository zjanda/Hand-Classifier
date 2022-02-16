from helpers import *
from tkinter.ttk import *
from multiprocessing import Process
from HandTrackingTesting import HandTrackingTesting
from HandTrackingDataCreator import HandTrackingDataCreator


# os.system('python3 HandTrackingDataCreator.py')
def runRecordData():
    p1 = Process(target=HandTrackingDataCreator)
    p1.start()


def runTrainModel():
    print(os.system('jupyter nbconvert --to script modeltrain.ipynb'))
    import modeltrain


def runTestModel():
    Process(target=HandTrackingTesting).start()

if __name__ == '__main__':
    mainWindow = Tk()

    app_name = 'Zack\'s Fantastic Hand Classifier'
    mainWindow.title(app_name)
    mainWindow.geometry('350x220')
    mainWindow.eval('tk::PlaceWindow . center')
    style = Style()
    style.configure('W.TButton', font=('calibri', 10, 'bold'))
    Label(mainWindow, text=f'Welcome to {app_name}').pack(pady=20)
    Button(mainWindow, text="Record Data", style='W.TButton', command=runRecordData).pack(pady=10)
    Button(mainWindow, text="Train Model", style='W.TButton', command=runTrainModel).pack(pady=10)
    Button(mainWindow, text="Test Model", style='W.TButton', command=runTestModel).pack(pady=10)

    mainloop()
