#-- coding: utf-8 --
import cv2
import tensorflow as tf
from Vgg300Net import Vgg300Net
import numpy as np
class SSD300Net(tf.keras.Model):
      #conv4 ==> 38 x 38
      # conv7 ==> 19 x 19
      # conv8 ==> 10 x 10
      # conv9 ==> 5 x 5
      # conv10 ==> 3 x 3
      # conv11 ==> 1 x 1
    def __init__(self, n_classes=21,
                 anchor_sizes=[ (21., 45.),
                                (45., 99.),
                                (99., 153.),
                                (153., 207.),
                                (207., 261.),
                                (261., 315.)
                                ],
                 anchor_ratios=[ [2, .5],
                                 [2, .5, 3, 1./3],
                                 [2, .5, 3, 1./3],
                                 [2, .5, 3, 1./3],
                                 [2, .5],
                                 [2, .5]],
                 training = False
                 ):
        super(SSD300Net, self).__init__()
        tf.keras.backend.set_learning_phase(training)
        self.n_classes = n_classes
        self.anchor_sizes = anchor_sizes,
        self.anchor_ratios = anchor_ratios,
        self.bodyNet = Vgg300Net()
        #num_anchors = len(sizes) + len(ratios)
        self.box_conv_stage_4 = tf.keras.layers.Conv2D((len(anchor_sizes[0]) + len(anchor_ratios[0])) * 4, 3, padding='same')
        self.box_conv_stage_7 = tf.keras.layers.Conv2D((len(anchor_sizes[1]) + len(anchor_ratios[1])) * 4, 3, padding='same')
        self.box_conv_stage_8 = tf.keras.layers.Conv2D((len(anchor_sizes[2]) + len(anchor_ratios[2])) * 4, 3, padding='same')
        self.box_conv_stage_9 = tf.keras.layers.Conv2D((len(anchor_sizes[3]) + len(anchor_ratios[3])) * 4, 3, padding='same')
        self.box_conv_stage_10 = tf.keras.layers.Conv2D((len(anchor_sizes[4]) + len(anchor_ratios[4])) * 4, 3, padding='same')
        self.box_conv_stage_11 = tf.keras.layers.Conv2D((len(anchor_sizes[5]) + len(anchor_ratios[5])) * 4, 3, padding='same')
        self.box_layers = [self.box_conv_stage_4,
                           self.box_conv_stage_7,
                           self.box_conv_stage_8,
                           self.box_conv_stage_9,
                           self.box_conv_stage_10,
                           self.box_conv_stage_11]

        self.cls_conv_stage_4 = tf.keras.layers.Conv2D((len(anchor_sizes[0]) + len(anchor_ratios[0])) * n_classes, 3, padding='same')
        self.cls_conv_stage_7 = tf.keras.layers.Conv2D((len(anchor_sizes[1]) + len(anchor_ratios[1])) * n_classes, 3, padding='same')
        self.cls_conv_stage_8 = tf.keras.layers.Conv2D((len(anchor_sizes[2]) + len(anchor_ratios[2])) * n_classes, 3, padding='same')
        self.cls_conv_stage_9 = tf.keras.layers.Conv2D((len(anchor_sizes[3]) + len(anchor_ratios[3])) * n_classes, 3, padding='same')
        self.cls_conv_stage_10 = tf.keras.layers.Conv2D((len(anchor_sizes[4]) + len(anchor_ratios[4])) * n_classes, 3, padding='same')
        self.cls_conv_stage_11 = tf.keras.layers.Conv2D((len(anchor_sizes[5]) + len(anchor_ratios[5])) * n_classes, 3, padding='same')
        self.cls_layers = [self.cls_conv_stage_4,
                           self.cls_conv_stage_7,
                           self.cls_conv_stage_8,
                           self.cls_conv_stage_9,
                           self.cls_conv_stage_10,
                           self.cls_conv_stage_11]
        pass

    def call(self, input_tensor):
        #(batchsize,38,38,512)
        #(batchsize,19,19,1024)
        #(batchsize,10,10,512)
        #(batchsize,5,5,256)
        #(batchsize,3,3,256)
        #(batchsize,1,1,256)
        stage_results = self.bodyNet(input_tensor)
        softmax_cls_preds, loc_preds, cls_preds = [],[],[]
        for i, res in enumerate(stage_results):
            pre_shape = res.get_shape().as_list()[0:-1]
            n_anchors = len(self.anchor_sizes[0][i]) + len(self.anchor_ratios[0][i])

            loc_pre_shape = pre_shape + [n_anchors, 4]
            cls_pre_shape = pre_shape + [n_anchors, self.n_classes]
            # numbers of anchors
            #n_anchors = len(self.sizes) + len(self.ratios)
            # location predictions
            loc_pred = self.box_layers[i](res)
            loc_pred = tf.reshape(loc_pred, loc_pre_shape)
            # class prediction
            cls_pred = self.cls_layers[i](res)
            cls_pred = tf.reshape(cls_pred,cls_pre_shape)
            cls_pred_ = tf.nn.softmax(cls_pred)
            softmax_cls_preds.append(cls_pred_)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
            pass
        return loc_preds, cls_preds, softmax_cls_preds
        #loc_preds.shape=[(2,38,38,4,4),(2,19,19,6,4),(2,10,10,6,4),(2,5,5,6,4),(2,3,3,4,4),(2,1,1,4,4)]
        #cls_preds.shape=[(2,38,38,4,21),(2,19,19,6,21),(2,10,10,6,21),(2,5,5,6,21),(2,3,3,4,21),(2,1,1,4,21)]
        pass


        def predict(self, input_tensor):
            loc_preds, cls_preds, softmax_cls_preds = self.call(input_tensor)
            return loc_preds, cls_preds, softmax_cls_preds
            pass


        #Compute the default anchor boxes, given an image shape.
        # def detect(self, image, anchors, net_size=(300,300)):
        #     image_h,image_w = image.shape
        #     preprocess_img = cv2.resize(image/255.,net_size)
        #     new_image = np.expand_dims(preprocess_img,axis=0)
        #     ys = self.predict(new_image)
        #     scores, bboxs =
        #
        #
        #     pass
        #
        # def bboxes_select(self, softmax_cls_preds, loc_preds, select_threshold=None, num_classes=21):
        #     l_scores = []
        #     l_bboxes = []
        #     cls_shape = softmax_cls_preds.shape
        #     loc_shape = loc_preds.shape
        #     for i in range(len(softmax_cls_preds)):
        #         scores, bboxes =
        #
        #         pass
        #     pass

    pass

if __name__ == "__main__":
    input = np.random.randn(2,300,300,3)
    x = tf.constant(input)
    print(x.shape)
    print(x.get_shape().as_list()[1:-1])
    model = SSD300Net()
    loc_preds, cls_preds, softmax_cls_preds = model(input)
    # loc_preds[0]: (2, 38, 38, 4, 4)
    # loc_preds[1]: (2, 19, 19, 6, 4)
    # loc_preds[2]: (2, 10, 10, 6, 4)
    # loc_preds[3]: (2, 5, 5, 6, 4)
    # loc_preds[4]: (2, 3, 3, 4, 4)
    # loc_preds[5]: (2, 1, 1, 4, 4)
    # cls_preds[0]: (2, 38, 38, 4, 21)
    # cls_preds[1]: (2, 19, 19, 6, 21)
    # cls_preds[2]: (2, 10, 10, 6, 21)
    # cls_preds[3]: (2, 5, 5, 6, 21)
    # cls_preds[4]: (2, 3, 3, 4, 21)
    # cls_preds[5]: (2, 1, 1, 4, 21)
    print(f"loc_preds:{loc_preds[0].shape}")
    print(f"loc_preds:{loc_preds[1].shape}")
    print(f"loc_preds:{loc_preds[2].shape}")
    print(f"loc_preds:{loc_preds[3].shape}")
    print(f"loc_preds:{loc_preds[4].shape}")
    print(f"loc_preds:{loc_preds[5].shape}")
    print(f"cls_preds:{cls_preds[0].shape}")
    print(f"cls_preds:{cls_preds[1].shape}")
    print(f"cls_preds:{cls_preds[2].shape}")
    print(f"cls_preds:{cls_preds[3].shape}")
    print(f"cls_preds:{cls_preds[4].shape}")
    print(f"cls_preds:{cls_preds[5].shape}")
    pass
