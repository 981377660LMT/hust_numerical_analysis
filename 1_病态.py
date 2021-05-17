# -*- coding: utf-8 -*-
from typing import List
import numpy as np
import json


# 注意poly和roots函数都是次数从高到低排序
# x^19系数对应干扰参数数组disturbution的第二项
def cal_root(disturbution: List[complex]) -> List[complex]:
    assert len(disturbution) == 21
    # 添加扰动的项
    disturbution_index = [i for i, j in enumerate(disturbution[::-1]) if j != 0]
    # 扰动转复数形式
    disturbution = [complex(str(num).replace(" ", "")) for num in disturbution]

    raw_polynomial_coefficients = np.poly([i for i in range(20, 0, -1)])
    disturbed_polynomial_coefficients = raw_polynomial_coefficients[:]
    disturbed_polynomial_coefficients = [
        disturbed_polynomial_coefficients[i] + disturbution[i] for i in range(21)
    ]
    disturbed_polynomial_roots = np.roots(disturbed_polynomial_coefficients)

    title_fragment = '_'.join(map(str, disturbution_index))
    with open(f'微扰第{title_fragment}项系数.json', 'w') as f:
        f.write(to_json(disturbed_polynomial_roots))

    return disturbed_polynomial_roots


# np.ndarray转json字符串
def to_json(complex_array: np.ndarray):
    array = [str(i) for i in complex_array]
    reversed_dict = {i: array[::-1][i - 1] for i in range(20, 0, -1)}
    return json.dumps(reversed_dict)


if __name__ == '__main__':
    for i in range(20, -1, -1):
        # 指定微扰的系数与大小
        d = {i: 0.0001 + 0.0001j}
        disturbution = [d.get(j, 0j) for j in range(20, -1, -1)]
        cal_root(disturbution)
