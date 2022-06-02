import cv2
import numpy as np


# 读取图片获得图片列表和轮廓列表
def process_list(_root, _num1, _num2):
    _list_img = []  # opencv读取的图片
    _list_cnt = []  # 排序过的轮廓
    _list_cnt_no = []  # 没有排序过的轮廓
    for _i in range(_num1, _num2):
        if _i < 10:
            _j = '0000' + str(_i)
        elif _i < 100:
            _j = '000' + str(_i)
        elif _i < 1000:
            _j = '00' + str(_i)
        else:
            _j = '0' + str(_i)
        # img = cv2.imread(_root + 'small_big_0723/' + _j + '.bmp')
        img = cv2.imread(_root + '50%_2/' + _j + '.bmp')
        ret, binary = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)
        _list_img.append(erosion)
        gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _list_cnt_no.append(contours)
        # bboxes = [cv2.boundingRect(_cnt) for _cnt in contours]
        # (contours, boundingBoxes) = zip(*sorted(zip(contours, bboxes), key=lambda b: b[1][1], reverse=False))
        # _list_cnt.append(contours)
    return _list_img, _list_cnt_no


# 计算轮廓面积
def cnt_area(_cnt):
    area = cv2.contourArea(_cnt)
    return area


# 获取轮廓数目列表和轮廓坐标（x,y,x1,y1）列表
def cnt_num(_list_cnt):
    _list_num = []
    _list_loc = []
    for _cnts in _list_cnt:
        num = 0
        for _cnt in _cnts:
            if cnt_area(_cnt) < 100:
                continue
            _x, _y, _w, _h = cv2.boundingRect(_cnt)
            _x1 = _x + _w
            _y1 = _y + _h
            _list_loc.append([_x, _y, _x1, _y1])
            num += 1
        _list_num.append(num)
    return _list_num, _list_loc


def save(_root, _i, _list_img):
    if _i < 10:
        _j = '0000' + str(_i)
    elif _i < 100:
        _j = '000' + str(_i)
    elif _i < 1000:
        _j = '00' + str(_i)
    else:
        _j = '0' + str(_i)
    cv2.imwrite(_root + _j + '.bmp', _list_img[_i])


def show(_list_img, _i):
    cv2.imshow("result", _list_img[_i])
    cv2.waitKey(100)
