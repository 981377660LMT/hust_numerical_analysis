# -*- coding: utf-8 -*-
import uuid
from typing import Callable, List, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# 用来正常显示中文标签
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False


def func_1(x: float):
    return 1 / (1 + 25 * pow(x, 2))


def func_2(x: float):
    return x / (1 + pow(x, 4))


def func_3(x: float):
    return np.arctan(x)


def cal_lagrange(data: List[Tuple[float, float]], x: float):
    """给定一组插值点，计算自变量x对应的插值多项式的值y

    Args:
        data (List[Tuple[float, float]]): 给定的插值点
        x (float): 给定的自变量

    Returns:
        float: 插值多项式对应的值
    """
    predict = 0.0
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    if x in data_x:
        # print "x is already known"
        return data_y[data_x.index(x)]
    for i in range(len(data_x)):
        # basis为拉格朗日基本多项式的值
        basis = 1
        for j in range(len(data_x)):
            if j != i:
                basis *= 1.0 * (x - data_x[j]) / (data_x[i] - data_x[j])
        predict += data_y[i] * basis
    return predict


def gen_data(start: float, end: float, n: int, *, raw_func: Callable):
    """给定函数和分点数，生成插值点"""
    x_interval = np.linspace(-1, 1, n + 1)
    y_value = [raw_func(x) for x in x_interval]
    return list(zip(x_interval, y_value))


def plot(
    start: float,
    end: float,
    n: int,
    *,
    data: List[Tuple[float, float]],
    raw_func: Callable,
    num=1,
    title='',
    type=1,
):
    """"给定原函数、插值点和横坐标的信息，画出图像"""
    x_interval = np.linspace(start, end, n)
    lagrange_value = [cal_lagrange(data, x) for x in x_interval]
    raw_value = [raw_func(x) for x in x_interval]

    plt.figure(num=num, figsize=(8, 5))
    (lagrange_line,) = plt.plot(x_interval, lagrange_value)
    (raw_line,) = plt.plot(x_interval, raw_value)

    plt.title(label=title, fontsize=24, fontweight='bold', pad=10)
    plt.legend(handles=[lagrange_line, raw_line], labels=['插值多项式函数', '原函数'], loc='best')
    plt.savefig(f'函数{type}_分点数{num}_{uuid.uuid4().hex}.jpg')
    plt.show()


def run(start: float, end: float, n: int, *, raw_func: Callable, type=1):
    """"给定原函数，画出分点数目2到10的图像"""
    for num in range(2, 3):
        data = gen_data(start, end, num, raw_func=raw_func)
        plot(
            start,
            end,
            n,
            data=data,
            raw_func=raw_func,
            title=f'函数{type}分点数为{num}的情况',
            num=num,
            type=type,
        )


if __name__ == '__main__':
    run(-1, 1, 300, raw_func=func_1, type=1)
    run(-5, 5, 300, raw_func=func_2, type=2)
    run(-5, 5, 300, raw_func=func_3, type=3)
