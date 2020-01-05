# -*- coding: utf-8 -*-
# @Time    : 2020/1/2 0:58
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : demo.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


def non_max_suppression():
    predictions_with_boxes = np.random.random((8732,25))
    predictions_part = predictions_with_boxes[:, 4:]  # 单张图片检测
    idxs = tf.argmax(predictions_part, -1)
    obj_mask = tf.where(idxs == 0, 0, 1)
    obj_mask = tf.expand_dims(obj_mask, -1)
    predictions_with_boxes = predictions_with_boxes * tf.cast(obj_mask, dtype=tf.float32)
    predictions_with_boxes = predictions_with_boxes.numpy()
    non_zero_idxs = np.nonzero(predictions_with_boxes)
    non_zero_idx = list(set(non_zero_idxs[0]))
    predictions_with_boxes = predictions_with_boxes[non_zero_idx]
    print(predictions_with_boxes.shape)

    # 将概率小于阈值confidence_threshold的bbox的概率置为0
    predictions_part1 = predictions_with_boxes[:, 0:4]
    predictions_part2 = predictions_with_boxes[:, 4:]
    conf_mask = tf.where(predictions_part2 > 0.5, 1, 0)
    #conf_mask = tf.expand_dims(conf_mask, -1)
    predictions_part2 = tf.cast(predictions_part2,dtype=tf.float32) * tf.cast(conf_mask,dtype=tf.float32)
    predictions = tf.concat([predictions_part1, predictions_part2], -1)
    predictions = predictions.numpy()
    result = {}

    bbox_attrs = predictions[:, 0:4]
    bbox_classes = predictions[:, 4:]
    classes = np.argmax(bbox_classes, axis=-1)

    unique_classes = list(set(classes.reshape(-1)))

    for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
        cls_probs = bbox_classes[np.nonzero(cls_mask)][:, cls]
        cls_probs = np.expand_dims(cls_probs, -1)
        cls_boxes = np.concatenate((cls_boxes,cls_probs),-1)
        cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]

        cls_scores = cls_boxes[:, -1]
        cls_boxes = cls_boxes[:, :-1]

        while len(cls_boxes) > 0:
            box = cls_boxes[0]
            score = cls_scores[0]
            if not cls in result:
                result[cls] = []
            result[cls].append((box, score))

    return result


if __name__ == '__main__':
    #non_max_suppression()
    print(list(range(1,2)))
    pass