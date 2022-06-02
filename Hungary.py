from scipy.optimize import linear_sum_assignment
import numpy as np


# 计算IOU
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


# 匈牙利算法
def hungary(_list_loc_temp, _list_pre_temp):
    # 代价矩阵
    _dis_mat = np.zeros((3, 3), dtype=np.float32)
    _dis_mat[:] = 1.0
    # IOU矩阵
    _iou_mat = np.zeros((3, 3), dtype=np.float32)
    _list_loc_all = []
    _list_pre_all = []
    for k in range(0, int(len(_list_loc_temp)/2)):
        _list_pre_all.append(list(_list_pre_temp[2*k])+list(_list_pre_temp[2*k+1]))
        _list_loc_all.append(_list_loc_temp[2*k]+_list_loc_temp[2*k+1])
    for xy in range(0, len(_list_loc_all)):  # 更新cost矩阵
        for pre in range(0, len(_list_pre_all)):
            _dis_mat[xy][pre] = 1-cal_iou(_list_loc_all[xy], _list_pre_all[pre])
            _iou_mat[xy][pre] = cal_iou(_list_loc_all[xy], _list_pre_all[pre])
    row_ind, col_ind = linear_sum_assignment(_dis_mat)  # 匈牙利算法，得到行列索引。行：目标索引；列：kal索引
    # print("最优行索引：", row_ind)
    # print("最优列索引：", col_ind)
    return row_ind, col_ind, _iou_mat
