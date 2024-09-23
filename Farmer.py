import keyboard
import mouse
import time
import pyautogui as pg
from win32gui import FindWindow, GetWindowRect,MoveWindow
from OpenCV import get_monitor,CV2_detect,get_screenshot,get_detector
#0为找怪 1为战斗
state=0
def get_state(img):
    x=456
    y=123
    if(img[x,y]==[]):
        state=1
    else:
        state=0
def pressKey(key,t):
    keyboard.press(key)
    time.sleep(t)
    keyboard.release(key)



def HP_reveal(HP):
    if(HP<=30):
        pressKey('F3', 0.5)

def MP_reveal(MP):
    if(MP<=30):
        pressKey('F3', 0.5)


if __name__ == "__main__":
    window_name = "DEFAULT | Flyff Universe"
    window_class = 'Window Class'
    id = FindWindow(window_class, window_name)
    MoveWindow(id, 0, 0, 1080, 720, True)
    while True:
        img = get_screenshot()
        get_state(img)
        if(state==0):#找怪逻辑
            bbox = get_detector(img)
            time.sleep(1)
            if bbox.shape[0] != 0:
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x_,y_=(x0+x1)/2,(y0+y1)/2+50
                pg.doubleClick(x=x_, y=y_, duration=0, button='left')
            else:
                pressKey('left', 0.2)
                time.sleep(0.2)
                pressKey('w', 0.2)
                time.sleep(0.2)
        else:
            HP_reveal(30)
            MP_reveal(30)
            pressKey('F2', 0.5)
