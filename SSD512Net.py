#-- coding: utf-8 --
import tensorflow as tf
from Vgg512Net import Vgg512Net

class SSD512Net(tf.keras.Model):
    def __init__(self, n_classes, anchor_sizes, anchor_ratios):
        super(SSD512Net, self).__init__()
        self.n_classes = n_classes
        self.anchor_sizes = anchor_sizes,
        self.anchor_ratios = anchor_ratios,
        self.bodyNet = Vgg512Net()
        #num_anchors = len(sizes) + len(ratios)
        self.box_conv_stage_4 = tf.keras.layers.Conv2D((len(anchor_sizes[0]) + len(anchor_ratios[0])) * 4, 3)
        self.box_conv_stage_7 = tf.keras.layers.Conv2D((len(anchor_sizes[1]) + len(anchor_ratios[1])) * 4, 3)
        self.box_conv_stage_8 = tf.keras.layers.Conv2D((len(anchor_sizes[2]) + len(anchor_ratios[2])) * 4, 3)
        self.box_conv_stage_9 = tf.keras.layers.Conv2D((len(anchor_sizes[3]) + len(anchor_ratios[3])) * 4, 3)
        self.box_conv_stage_10 = tf.keras.layers.Conv2D((len(anchor_sizes[4]) + len(anchor_ratios[4])) * 4, 3)
        self.box_conv_stage_11 = tf.keras.layers.Conv2D((len(anchor_sizes[5]) + len(anchor_ratios[5])) * 4, 3)
        self.box_conv_stage_12 = tf.keras.layers.Conv2D((len(anchor_sizes[6]) + len(anchor_ratios[6])) * 4, 3)
        self.box_layers = [self.box_conv_stage_4,
                           self.box_conv_stage_7,
                           self.box_conv_stage_8,
                           self.box_conv_stage_9,
                           self.box_conv_stage_10,
                           self.box_conv_stage_11,
                           self.box_conv_stage_12]

        self.cls_conv_stage_4 = tf.keras.layers.Conv2D((len(anchor_sizes[0]) + len(anchor_ratios[0])) * n_classes, 3)
        self.cls_conv_stage_7 = tf.keras.layers.Conv2D((len(anchor_sizes[1]) + len(anchor_ratios[1])) * n_classes, 3)
        self.cls_conv_stage_8 = tf.keras.layers.Conv2D((len(anchor_sizes[2]) + len(anchor_ratios[2])) * n_classes, 3)
        self.cls_conv_stage_9 = tf.keras.layers.Conv2D((len(anchor_sizes[3]) + len(anchor_ratios[3])) * n_classes, 3)
        self.cls_conv_stage_10 = tf.keras.layers.Conv2D((len(anchor_sizes[4]) + len(anchor_ratios[4])) * n_classes, 3)
        self.cls_conv_stage_11 = tf.keras.layers.Conv2D((len(anchor_sizes[5]) + len(anchor_ratios[5])) * n_classes, 3)
        self.cls_conv_stage_11 = tf.keras.layers.Conv2D((len(anchor_sizes[6]) + len(anchor_ratios[6])) * n_classes, 3)
        self.cls_layers = [self.cls_conv_stage_4,
                           self.cls_conv_stage_7,
                           self.cls_conv_stage_8,
                           self.cls_conv_stage_9,
                           self.cls_conv_stage_10,
                           self.cls_conv_stage_11,
                           self.cls_conv_stage_12]
        pass

    def call(self, input_tensor):
        stage_results = self.bodyNet(input_tensor)
        softmax_cls_preds, loc_preds, cls_preds = [],[],[]
        for i, res in enumerate(stage_results):
            pre_shape = res.get_shape().as_list()[1:-1]
            pre_shape = [-1] + pre_shape

            # numbers of anchors
            #n_anchors = len(self.sizes) + len(self.ratios)
            # location predictions
            loc_pred = self.box_layers[i](res)
            loc_pred = tf.reshape(loc_pred, pre_shape + [(len(self.anchor_sizes[i]) + len(self.anchor_ratios[i])), 4])
            # class prediction
            cls_pred = self.cls_layers[i](res)
            cls_pred = tf.reshape(cls_pred, pre_shape + [(len(self.anchor_sizes[i]) + len(self.anchor_ratios[i])), self.n_classes])
            cls_pred_ = tf.nn.softmax(cls_pred)
            softmax_cls_preds.append(cls_pred_)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
            pass
        return loc_preds, cls_preds, softmax_cls_preds
        pass
    pass

if __name__ == "__main__":
    SSD512Net()
    pass
