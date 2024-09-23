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
        Non-Maximum Suppression��NMS���Ǽ���ֵ���ơ���������˼��⣬������Щ�Ǽ���ֵ��Ԫ�أ���������ֵԪ�ء�����Ҫ����Ŀ���⣬Ŀ����٣�3D�ؽ��������ھ�ȡ�
        ���ص���ѡ���ȥ��
    """
    # x1��y1��x2��y2���Լ�score��ֵ
    # ��x1��y1����x2��y2��Ϊbox�����Ϻ����½Ǳ�
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # ÿһ����ѡ������
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order�ǰ���score���������
    order = scores.argsort()[::-1]
    # print("order:",order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # ���㵱ǰ���������ο����������ο���ཻ������꣬���õ�numpy��broadcast���ƣ��õ���������
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # �����ཻ������,ע����ο��ཻʱw��h��������Ǹ�������0����
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # �����ص���IOU���ص����/�����1+���2-�ص������
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # �ҵ��ص��Ȳ�������ֵ�ľ��ο�����
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # ��order���и��£�����ǰ��õ��ľ��ο�����Ҫ�Ⱦ��ο���ԭorder�����е�����С1������Ҫ�����1�ӻ���
        order = order[inds + 1]
    return keep


def template(img_gray, template_img, template_threshold=0.5):
    '''
        ģ�庯������������ģ�壬����ƥ�䵽�����м���
        ����Ϊһ���б� ����ƥ�䵽�ľ��ε����Ͻ����꣨xmin,ymin)+���½����꣨xmax,ymax��
        img_gray:�����ĻҶ�ͼƬ��ʽ
        template_img:ģ��Сͼ��Ҳ�ǻҶȻ���
        template_threshold:ģ��ƥ������Ŷ�
    '''

    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    start_time = time.time()
    loc = np.where(res >= template_threshold)  # ����ģ����ֵ��Ŀ������
    score = res[res >= template_threshold]  # ����ģ����ֵ��Ŀ�����Ŷ�
    # ��ģ������������д�������Ͻǡ����½ǵĸ�ʽ
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1)  # ���n��1��ά��
    xmax = xmax.reshape(-1, 1)  # ���n��1��ά��
    ymax = ymax.reshape(-1, 1)  # ���n��1��ά��
    ymin = ymin.reshape(-1, 1)  # ���n��1��ά��
    score = score.reshape(-1, 1)  # ���n��1��ά��
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  # ��xmin��ymin��xmax��yamx��scores�����н���ƴ��
    thresh = 0.3  # NMS�����IOU��������ֵ
    keep_dets = py_nms(data_hstack, thresh)
    print("nms time:", time.time() - start_time)  # ��ӡ���ݴ���nms����ʱ��
    dets = data_hstack[keep_dets]  # ���յ�nms��õľ��ο�
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
        �ú���Ϊ��ȡ��������·���������ļ�����ͼƬ����
        ����·�� ����image
    '''
    stream = open(path, 'rb')
    bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

def CV2_detect():
    global img_src
    while True:
        img=img_src.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ת���ɻ�ɫ
        template_img = cv2.imread('temp.jpg', 0)  # ģ��Сͼ
        template_threshold = 0.8  # ģ�����Ŷ�
        bboxes = template(img_gray, template_img, template_threshold)
        for coord in bboxes:
            cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)

        x,y=480,40
        cv2.rectangle(img, (int(x), int(y)), (int(x+1), int(y+1)), (192, 192, 192), 3)
        cv2.putText(img, 'Fight_State', (max(0, x), max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),  2)
        print(img[480,40])
        if  np.array_equal(img[480,40], np.array([27,142,137])):
            print('����ս��״̬')
        cv2.rectangle(img, (150, 40), (151, 41), (192, 192, 192), 3)
        cv2.putText(img, 'HP_State', (max(0, 140), max(0, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        print("HP״̬��" + str(img[150, 40,0]))
        cv2.rectangle(img, (130, 60), (131, 61), (192, 192, 192), 3)
        cv2.putText(img, 'MP_State', (max(0, 130), max(0, 61 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        print("MP״̬��" + str(img[130, 60,0]))



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
    ���β���ʹ�õ��߳�+opencv���ٽ���+ģ��ƥ��
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
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)  # ת���ɻ�ɫ
    template_img = cv2.imread('temp.jpg', 0)  # ģ��Сͼ
    template_threshold = 0.8  # ģ�����Ŷ�
    bboxes = template(img_gray, template_img, template_threshold)
    return get_largestBox(bboxes)