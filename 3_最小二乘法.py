# -*- coding: utf-8 -*-
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray, square
from scipy.optimize import leastsq
from sklearn.metrics import mean_squared_error

# 用来正常显示中文标签
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False


# 通过函数定义拟合函数的形式。
def fitted_func(klist: List[float], x: ndarray) -> ndarray:
    k3, k2, k1, k0 = klist
    return k3 * pow(x, 3) + k2 * pow(x, 2) + k1 * pow(x, 1) + k0 * pow(x, 0)


# 定义残差项
def error(klist: List[float], x: ndarray, y: float):
    return fitted_func(klist, x) - y


# 画出拟合后的曲线
def plot(
    x: ndarray,
    y: ndarray,
    x_interval: ndarray,
    y_fitted: ndarray,
    k3: str,
    k2: str,
    k1: str,
    k0: str,
):
    plt.figure()
    (origin_line,) = plt.plot(x, y, 'ro', label='原数据')
    (fitted_line,) = plt.plot(x_interval, y_fitted, '-b', label='拟合函数')
    coefficients = f'{k3}x^3{k2}x^2{k1}x^1{k0}'[1:]
    plt.title(label=f'拟合曲线为 {coefficients}', fontsize=12, fontweight='bold', pad=10)
    plt.legend(handles=[origin_line, fitted_line], labels=['原数据', '拟合函数'], loc='best')
    plt.show()


# 计算拟合曲线系数
def cal_coefficient(*, init_klist, x_data, y_data) -> List[float]:
    # leastsq已经将最小二乘法的”偏差平方”和”最小化”封装在函数中，所以偏差函数不用自行加“平方”
    return leastsq(error, init_klist, args=(x_data, y_data))[0]


# 计算最后的平方误差
def cal_square_error(*, y_true: ndarray, y_pred: ndarray):
    assert len(y_true) == len(y_pred)
    return mean_squared_error(y_true, y_pred) * len(y_true)


def run():
    x = np.array([-1, -0.5, 0, 0.5, 1, 1.5, 2])
    y = np.array([-4.447, -0.452, -0.551, 0.048, -0.447, 0.549, 4.552])
    result = cal_coefficient(init_klist=[0.01, 0.01, 0.01, 0.0], x_data=x, y_data=y)
    square_error = cal_square_error(y_true=y, y_pred=fitted_func(result, x))

    k3, k2, k1, k0 = map(lambda x: '+' + str(round(x, 3)) if x >= 0 else str(round(x, 3)), result)
    x_interval = np.linspace(-2, 2, 200)
    y_fitted = fitted_func(result, x_interval)
    plot(
        x,
        y,
        x_interval,
        y_fitted,
        k3,
        k2,
        k1,
        k0,
    )

    print('拟合系数为', result)
    print('平方误差为', square_error)


if __name__ == '__main__':
    run()
