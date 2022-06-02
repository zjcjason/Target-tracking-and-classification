import numpy as np
import data
from kalman import KalmanFilter
import cv2
from Hungary import hungary
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


# 新出现目标，初始化kf、预测点和真是坐标
def create(_list_kf, _list_num, _list_loc, _i, _j, _list_pre_temp, _list_loc_temp):
    if (len(_list_kf) / 2) < _list_num[_i]:  # 初始化卡尔曼滤波器和参数
        kf = KalmanFilter()
        _list_kf.append(kf)
        kf = KalmanFilter()
        _list_kf.append(kf)
    if (len(_list_pre_temp) / 2) < _list_num[_i]:
        _list_pre_temp.append([0, 0])
        _list_pre_temp.append([0, 0])
    if (len(_list_loc_temp) / 2) < _list_num[_i]:
        _list_loc_temp.append(_list_loc[_j][:2])
        _list_loc_temp.append(_list_loc[_j][2:])
    return _list_kf, _list_loc_temp, _list_pre_temp


def draw_real(_list_img, _list_loc, _list_loc_temp, _i, _j, _k):
    _list_loc_temp[2 * _k] = _list_loc[_j][:2]
    _list_loc_temp[2 * _k + 1] = _list_loc[_j][2:]
    cv2.rectangle(_list_img[_i], _list_loc[_j][:2], _list_loc[_j][2:], (0, 0, 255), 2)


def list_to_array(_list_pre_temp, _list_loc_temp):
    array_pre_temp_xy = np.array(_list_pre_temp[::2])  # 取出所有xy值
    array_pre_temp_x1y1 = np.array(_list_pre_temp[1::2])  # 取出所有x1y1值
    list_loc_temp_xy = _list_loc_temp[::2]
    list_loc_temp_x1y1 = _list_loc_temp[1::2]
    array_loc_temp_xy = np.array(list_loc_temp_xy)
    array_loc_temp_x1y1 = np.array(list_loc_temp_x1y1)
    array_temp_xy = np.append(array_loc_temp_xy, array_pre_temp_xy, axis=0)
    array_temp_x1y1 = np.append(array_loc_temp_x1y1, array_pre_temp_x1y1, axis=0)
    array_total = np.append(np.array(_list_loc_temp), np.array(_list_pre_temp), axis=0)
    return array_temp_xy, array_temp_x1y1, array_total


def kf_pre_xy(_row_ind_xy, _col_ind_xy, _list_kf, _list_loc_temp, _list_pre_temp):
    for ind in range(len(_row_ind_xy)):
        if ind >= len(_list_loc_temp) / 2:
            continue
        _list_pre_temp[2 * _col_ind_xy[ind]] = _list_kf[2 * _col_ind_xy[ind]].predict(
            _list_loc_temp[2 * _row_ind_xy[ind]])


def kf_pre_x1y1(_row_ind_x1y1, _col_ind_x1y1, _list_kf, _list_loc_temp, _list_pre_temp):
    for ind in range(len(_row_ind_x1y1)):
        if ind >= len(_list_loc_temp) / 2:
            continue
        _list_pre_temp[2 * _col_ind_x1y1[ind] + 1] = _list_kf[2 * _col_ind_x1y1[ind] + 1].predict(
            _list_loc_temp[2 * _row_ind_x1y1[ind] + 1])


# 当目标消失超过5帧，将对应的预测框置0，kf重新初始化
def life(_time_2, _time_3, _list_pre_temp, _list_kf):
    if _time_2 > 5 & len(_list_pre_temp) > 2:
        kf = KalmanFilter()
        _list_kf[2] = kf
        kf = KalmanFilter()
        _list_kf[3] = kf
        _list_pre_temp[2] = [0, 0]
        _list_pre_temp[3] = [0, 0]
    if _time_3 > 5 & len(_list_pre_temp) > 4:
        kf = KalmanFilter()
        _list_kf[4] = kf
        kf = KalmanFilter()
        _list_kf[5] = kf
        _list_pre_temp[4] = [0, 0]
        _list_pre_temp[5] = [0, 0]


def draw_pre(_list_img, _col_ind, _i, _list_pre_temp, _list_loc_temp):
    # print("col_ind:", _col_ind)
    # print("list_pre_temp:", _list_pre_temp)
    for draw in _col_ind:
        if 2 * draw >= len(_list_loc_temp):
            continue
        if draw == 0:
            cv2.putText(_list_img[_i], "1", (_list_pre_temp[2 * draw + 1][0], _list_pre_temp[2 * draw + 1][1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        if draw == 1:
            cv2.putText(_list_img[_i], "2", (_list_pre_temp[2 * draw + 1][0], _list_pre_temp[2 * draw + 1][1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        if draw == 2:
            cv2.putText(_list_img[_i], "3", (_list_pre_temp[2 * draw + 1][0], _list_pre_temp[2 * draw + 1][1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(_list_img[_i], (_list_pre_temp[2 * draw][0], _list_pre_temp[2 * draw][1]),
                      (_list_pre_temp[2 * draw + 1][0], _list_pre_temp[2 * draw + 1][1]), (255, 0, 0), 2)


def main():
    root = './dataset/'
    list_kf = []
    list_img, list_cnt = data.process_list(root, 0, 50)  # 获得所有的原图列表和轮廓列表
    list_num, list_loc = data.cnt_num(list_cnt)  # 获得轮廓数量以及轮廓坐标列表[x,y,x1,y1]
    list_pre_temp = []  # 预测结果
    list_iou = []  # iou计算结果
    i = 0  # 原图计数、画图计数
    j = 0  # 轮廓计数
    time_2 = 0
    time_3 = 0
    for cnts in list_cnt:
        print("第", i, "轮")
        list_loc_temp = []  # 一组图片轮廓坐标
        k = 0  # 单张图的轮廓数
        iou_sum = 0
        for cnt in cnts:
            if data.cnt_area(cnt) < 100:
                continue
            list_kf, list_loc_temp, list_pre_temp = \
                create(list_kf, list_num, list_loc, i, j, list_pre_temp, list_loc_temp)
            draw_real(list_img, list_loc, list_loc_temp, i, j, k)
            j += 1  # 统计轮廓数后给新目标的list_loc_temp初始化以及画轮廓
            k += 1  # 统计单张图的轮廓数后给list_pre_temp赋坐标
        # 以上代码完成了新出现目标后list_loc_temp,list_pre_temp的初始化，以及画出一张图中的实际轮廓
        row_ind, col_ind, iou_mat = hungary(list_loc_temp, list_pre_temp)
        for iou_ind in range(0, 3):
            if row_ind[iou_ind] > len(list_loc_temp) | col_ind[iou_ind] > len(list_loc_temp):
                continue
            iou_sum += iou_mat[row_ind[iou_ind]][col_ind[iou_ind]]
        if list_num[i] != 0:
            iou_single = iou_sum / list_num[i]
        if list_num[i] == 0 & i > 0:
            iou_single = list_iou[i - 1]
        list_iou.append(iou_single)
        kf_pre_xy(row_ind, col_ind, list_kf, list_loc_temp, list_pre_temp)  # 根据匈牙利算法的结果，对应的kf预测对应的点
        kf_pre_x1y1(row_ind, col_ind, list_kf, list_loc_temp, list_pre_temp)

        life(time_2, time_3, list_pre_temp, list_kf)  # 不保留KF
        draw_pre(list_img, col_ind, i, list_pre_temp, list_loc_temp)
        time_2 += 1
        time_3 += 1
        if list_num[i] > 1:
            time_2 = 0
        if list_num[i] > 2:
            time_3 = 0
        data.show(list_img, i)  # 展示结果
        # data.save(root+'new/', i, list_img)  #保存结果
        i += 1
        print("--------")
    # iou_avg = sum(list_iou) / 1500
    # list_iou_save = np.array(list_iou)
    # np.save('list_iou.npy', list_iou_save)
    # plt.plot(list_iou)
    # plt.title("1500帧结果")
    # plt.xlabel("帧数")
    # plt.ylabel("IOU")
    # plt.savefig('list_iou.jpg')
    # plt.show()
    # cv2.destroyAllWindows()  # 关闭所有显示窗口，和上文save成对使用


if __name__ == '__main__':
    main()
