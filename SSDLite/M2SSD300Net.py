#-- coding: utf-8 --
import collections
import tensorflow as tf
import numpy as np
from backbone import body_net

def detect_block(inputs,out_channel):
    x = tf.keras.layers.DepthwiseConv2D(3,1,padding='same',use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(out_channel,1,1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x
    pass


M2SSDNetParams = collections.namedtuple('M2SSDNetParams',['n_classes','confidence_thresh','iou_threshold','variances','anchor_sizes','anchor_ratios'])


def M2SSDNet(inputs,net_params):
    inputs = tf.keras.layers.Input(shape=inputs.shape[1:])
    stage_results = body_net(inputs)
    anchors = []
    for anchor_size, anchor_ratio in zip(net_params.anchor_sizes, net_params.anchor_ratios):
        feat_anchors = []
        feat_anchors.append([anchor_size[0], anchor_size[0]])
        for val in anchor_ratio:
            feat_anchors.append([anchor_size[0] * np.sqrt(val), anchor_size[0] / np.sqrt(val)])
            pass
        feat_anchors.append([anchor_size[1], anchor_size[1]])
        anchors.append(feat_anchors)
        pass

    # print(np.array(np.array(anchors)[0]).shape)
    # print(np.array(np.array(anchors)[1]).shape)
    # print(np.array(np.array(anchors)[2]).shape)
    # print(np.array(np.array(anchors)[3]).shape)
    # print(np.array(np.array(anchors)[4]).shape)
    # print(np.array(np.array(anchors)[5]).shape)

    # num_anchors = len(sizes) + len(ratios)
    loc_preds, cls_preds = [], [], []
    for i, res in enumerate(stage_results):
        pre_shape = res.get_shape().as_list()[0:-1]
        n_anchors = len(net_params.anchor_sizes[0][i]) + len(net_params.anchor_ratios[0][i])

        loc_pre_shape = pre_shape + [n_anchors, 4]
        cls_pre_shape = pre_shape + [n_anchors, net_params.n_classes]
        # numbers of anchors
        # n_anchors = len(self.sizes) + len(self.ratios)
        # location predictions
        loc_pred = detect_block(stage_results[i],(len(net_params.anchor_sizes[i]) + len(net_params.anchor_ratios[i])) * 4)
        loc_pred = tf.reshape(loc_pred, loc_pre_shape)
        # class prediction
        cls_pred = detect_block(stage_results[i],(len(net_params.anchor_sizes[i]) + len(net_params.anchor_ratios[i])) * net_params.n_classes)
        cls_pred = tf.reshape(cls_pred, cls_pre_shape)
        cls_pred = tf.nn.softmax(cls_pred)
        cls_preds.append(cls_pred)
        loc_preds.append(loc_pred)
        pass
    return loc_preds, cls_preds
    pass