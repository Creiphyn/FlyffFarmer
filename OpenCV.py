# coding=gbk
import numpy as np
import cv2
from PIL import ImageGrab
from win32gui import FindWindow, GetWindowRect,MoveWindow
import time
import os
import mss
import threading
img_src=np.zeros((1280,720,3),np.uint8)

def py_nms(dets, thresh):
    """
        Pure Python NMS baseline.
        Non-Maximum Suppression（NMS）非极大值抑制。从字面意思理解，抑制那些非极大值的元素，保留极大值元素。其主要用于目标检测，目标跟踪，3D重建，数据挖掘等。
        将重叠的选择框去除
    """
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]
    # print("order:",order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def template(img_gray, template_img, template_threshold=0.5):
    '''
        模板函数，根据输入模板，返回匹配到的所有监测框
        返回为一个列表 包括匹配到的矩形的左上角坐标（xmin,ymin)+右下角坐标（xmax,ymax）
        img_gray:待检测的灰度图片格式
        template_img:模板小图，也是灰度化了
        template_threshold:模板匹配的置信度
    '''

    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    start_time = time.time()
    loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标
    score = res[res >= template_threshold]  # 大于模板阈值的目标置信度
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接
    thresh = 0.3  # NMS里面的IOU交互比阈值
    keep_dets = py_nms(data_hstack, thresh)
    print("nms time:", time.time() - start_time)  # 打印数据处理到nms运行时间
    dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
    return dets


def get_largestBox(bboxes):
    largest = -1
    bbox_largest = np.array([])
    for bbox in bboxes:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        area = (x1 - x0) * (y1 - y0)
        if area > largest:
            largest = area
            bbox_largest = bbox
    return bbox_largest

def imread_unicode(path):
    '''
        该函数为读取包含中文路径或中文文件名的图片函数
        输入路径 返回image
    '''
    stream = open(path, 'rb')
    bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

def CV2_detect():
    global img_src
    while True:
        img=img_src.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化成灰色
        template_img = cv2.imread('temp.jpg', 0)  # 模板小图
        template_threshold = 0.8  # 模板置信度
        bboxes = template(img_gray, template_img, template_threshold)
        for coord in bboxes:
            cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)

        x,y=480,40
        cv2.rectangle(img, (int(x), int(y)), (int(x+1), int(y+1)), (192, 192, 192), 3)
        cv2.putText(img, 'Fight_State', (max(0, x), max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),  2)
        print(img[480,40])
        if  np.array_equal(img[480,40], np.array([27,142,137])):
            print('进入战斗状态')
        cv2.rectangle(img, (150, 40), (151, 41), (192, 192, 192), 3)
        cv2.putText(img, 'HP_State', (max(0, 140), max(0, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        print("HP状态：" + str(img[150, 40,0]))
        cv2.rectangle(img, (130, 60), (131, 61), (192, 192, 192), 3)
        cv2.putText(img, 'MP_State', (max(0, 130), max(0, 61 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        print("MP状态：" + str(img[130, 60,0]))



        cv2.imshow("", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        #return get_largestBox(bboxes)



def get_screenshot():

    window_name = "DEFAULT | Flyff Universe"
    window_class = 'Window Class'
    id = FindWindow(window_class, window_name)
    x0,y0,x1,y1 = GetWindowRect(id)
    monitor = {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}
    img_src = np.array(mss.mss().grab(monitor))
    img_src = img_src[:, :, :3]
    img_src = img_src[30:-10,10:-10]
    return img_src


def get_monitor():
    global img_src
    while True:
        #last_time = time.time()
        img_src = get_screenshot()
        #print("fps: {}".format(1 / (time.time() - last_time+0.000000001)))
        #cv2.imshow('screenshot', img_src)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
             #cv2.destroyAllWindows()
             #break

if __name__ == "__main__":
    window_name = "DEFAULT | Flyff Universe"
    window_class = 'Window Class'
    id = FindWindow(window_class, window_name)
    MoveWindow(id, 0, 0, 1080, 720, True)
    t1 = threading.Thread(target=get_monitor, args=())
    t1.start()
    t2 = threading.Thread(target=CV2_detect, args=())
    t2.start()

    '''
    初次测试使用单线程+opencv慢速截屏+模板匹配
    '''
    # while True:
    #     window_name = "DEFAULT | Flyff Universe"
    #     window_class='Window Class'
    #     id = FindWindow(window_class, window_name)
    #
    #     bbox = GetWindowRect(id)
    #     print(bbox)
    #     image_array = np.array(ImageGrab.grab(bbox=bbox))
    #     image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    #
    #     #image_array, boxes, positions, confidences, classIDs = CV2_detect(image_array)
    #     image_array= CV2_detect(image_array)
    #     cv2.imshow('screenshot', image_array)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
def get_detector(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)  # 转化成灰色
    template_img = cv2.imread('temp.jpg', 0)  # 模板小图
    template_threshold = 0.8  # 模板置信度
    bboxes = template(img_gray, template_img, template_threshold)
    return get_largestBox(bboxes)