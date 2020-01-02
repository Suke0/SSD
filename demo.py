# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 0:58
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : demo.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


def non_max_suppression():
    predictions_with_boxes = np.random.random((1,3,6))
    predictions_part = predictions_with_boxes[:, :, 4:]
    print(predictions_part)
    arr = np.argmax(predictions_part,-1)
    print(arr)
    a = np.where(arr == 0,False,True)
    print(a)

if __name__ == '__main__':
    non_max_suppression()
    pass