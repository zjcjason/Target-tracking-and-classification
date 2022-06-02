import cv2
import numpy as np


# 卡尔曼滤波
class KalmanFilter:
    def __init__(self):
        # 创建一个6个状态维度，2个测量维度的卡尔曼滤波
        self.kf = cv2.KalmanFilter(6, 2)
        dt = 1.8
        # 测量矩阵
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        # 转移矩阵
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0, 0.5 * dt * dt, 0], [0, 1, 0, dt, 0, 0.5 * dt * dt], [0, 0, 1, 0, dt, 0],
             [0, 0, 0, 1, 0, dt], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
        # 过程噪声矩阵
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
                                           np.float32) * 0.001

    def predict(self, _xy):
        _x, _y = _xy
        measured = np.array([[np.float32(_x)], [np.float32(_y)]])
        # 根据测量更新预测状态
        self.kf.correct(measured)
        # 预测
        predicted = self.kf.predict()
        _x, _y = int(predicted[0]), int(predicted[1])
        return _x, _y
