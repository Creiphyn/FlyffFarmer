# coding=gbk
import numpy as np
import cv2
from PIL import ImageGrab
from win32gui import FindWindow, GetWindowRect

while True:
    window_name = "明日之后 - MuMu模拟器"
    id = FindWindow(None, window_name)
    bbox = GetWindowRect(id)
    image_array = np.array(ImageGrab.grab(bbox=bbox))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    cv2.imshow('screenshot',image_array)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
