#-- coding: utf-8 --
import cv2
import tensorflow as tf
from Vgg300Net import Vgg300Net
import numpy as np
from PIL import ImageDraw,Image
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
                 # anchor_sizes=[(30., 60.),
                 #               (60., 111.),
                 #               (111., 162.),
                 #               (162., 213.),
                 #               (213., 264.),
                 #               (264., 315.)],
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
        #38*38特征图的4个先验框的宽高
        self.anchors = []

        for anchor_size,anchor_ratio in zip(anchor_sizes, anchor_ratios):
            feat_anchors = []
            feat_anchors.append([anchor_size[0], anchor_size[0]])
            for val in anchor_ratio:
                feat_anchors.append([anchor_size[0] * np.sqrt(val), anchor_size[0] / np.sqrt(val)])
                pass
            feat_anchors.append([anchor_size[1], anchor_size[1]])
            self.anchors.append(feat_anchors)
            pass

        print(np.array(np.array(self.anchors)[0]).shape)
        print(np.array(np.array(self.anchors)[1]).shape)
        print(np.array(np.array(self.anchors)[2]).shape)
        print(np.array(np.array(self.anchors)[3]).shape)
        print(np.array(np.array(self.anchors)[4]).shape)
        print(np.array(np.array(self.anchors)[5]).shape)

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
        #prediction4 = tf.concat([loc_preds[0], softmax_cls_preds[0]], -1)
        #prediction4 = tf.reshape(prediction4,(prediction4.shape[0], -1, prediction4.shape[-1]))
        #prediction7 = tf.concat([loc_preds[1], softmax_cls_preds[1]], -1)
        #prediction7 = tf.reshape(prediction7, (prediction7.shape[0], -1, prediction7.shape[-1]))
        #prediction8 = tf.concat([loc_preds[2], softmax_cls_preds[2]], -1)
        #prediction8 = tf.reshape(prediction8, (prediction8.shape[0], -1, prediction8.shape[-1]))
        #prediction9 = tf.concat([loc_preds[3], softmax_cls_preds[3]], -1)
        #prediction9 = tf.reshape(prediction9, (prediction9.shape[0], -1, prediction9.shape[-1]))
        #prediction10 = tf.concat([loc_preds[4], softmax_cls_preds[4]], -1)
        #prediction10 = tf.reshape(prediction10, (prediction10.shape[0], -1, prediction10.shape[-1]))
        #prediction11 = tf.concat([loc_preds[5], softmax_cls_preds[5]], -1)
        #prediction11 = tf.reshape(prediction11, (prediction11.shape[0], -1, prediction11.shape[-1]))
        #predictions = tf.concat([prediction4,prediction7,prediction8,prediction9,prediction10,prediction11],1)
        #print(predictions.shape)
        return loc_preds, softmax_cls_preds
        pass

    def detect(self,new_img,net_size=(300,300)):
        #image = cv2.imread(img_name)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image_h,image_w,_=image.shape
        #preprocess_img = cv2.resize(image/255.,net_size)
        #new_img = np.expand_dims(preprocess_img,axis=0)
        loc_preds, cls_preds = self.predict(new_img)
        loc_preds_ = []
        for pred, anchor in zip(loc_preds, self.anchors):
            loc_preds_.append(self.bbox_converse(pred, anchor, net_size))
            pass
        preds  = []
        for loc_pred, cls_pred in zip(loc_preds_, cls_preds):
            pred = tf.concat([loc_pred,cls_pred],-1)
            pred = tf.reshape(pred,[pred.shape[0],-1,pred.shape[-1]])
            print(f'pred.shape:{pred.shape}')
            # pred.shape: (2, 5776, 25)
            # pred.shape: (2, 2166, 25)
            # pred.shape: (2, 600, 25)
            # pred.shape: (2, 150, 25)
            # pred.shape: (2, 36, 25)
            # pred.shape: (2, 4, 25)
            preds.append(pred)
            pass
        predictions = tf.concat(preds,1)
        detected_boxes = self.detections_boxes(predictions,net_size)
        print(f'predictions.shape:{predictions.shape}')
        #predictions.shape:(2, 8732, 25)
        conf_threshold = 0.5  # 置信度阈值
        iou_threshold = 0.4  # 重叠区域阈值
        filtered_boxes = self.non_max_suppression(detected_boxes, confidence_threshold=conf_threshold, iou_threshold=iou_threshold)
        return filtered_boxes
        pass

    def bbox_converse(self, pred, anchor, net_size):
        #pred : (2,38,38,4,4)
        center_x_y, w_h = tf.split(pred, [2, 2], axis=-1)
        grid_x = range(0,center_x_y.shape[1])
        grid_y = range(0,center_x_y.shape[2])
        a, b = tf.meshgrid(grid_x,grid_y)
        x_offset = tf.reshape(a,(-1,1))
        y_offset = tf.reshape(b,(-1,1))
        x_y_offset = tf.concat([x_offset,y_offset],-1)
        x_y_offset = tf.tile(x_y_offset,[center_x_y.shape[0],center_x_y.shape[-2]])
        x_y_offset = tf.reshape(x_y_offset,center_x_y.shape)
        anchor_wh = tf.tile(anchor, [np.prod(center_x_y.shape[0:3]), 1])
        anchor_wh = tf.reshape(anchor_wh, w_h.shape)
        center_x_y = center_x_y * anchor_wh + tf.cast(x_y_offset,tf.float32)
        stride = net_size[0] / center_x_y.shape[1]
        center_x_y = center_x_y * stride

        w_h = anchor_wh * tf.exp(w_h)
        pred = tf.concat([center_x_y,w_h],-1)
        return pred
        pass

    def detections_boxes(self, detections,net_size):
        center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = np.maximum(np.minimum(center_x - w2, net_size[0]), 0)
        y0 = np.maximum(np.minimum(center_y - h2, net_size[1]), 0)
        x1 = np.maximum(np.minimum(center_x + w2, net_size[0]), 0)
        y1 = np.maximum(np.minimum(center_y + h2, net_size[1]), 0)

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detections = tf.concat([boxes, attrs], axis=-1)
        return detections

    # 定义函数计算两个框的内部重叠情况（IOU）box1，box2为左上、右下的坐标[x0, y0, x1, x2]
    def iou(self, box1, box2):

        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        # 分母加个1e-05，避免除数为 0
        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou

    # 使用NMS方法，对结果去重
    def non_max_suppression(self, predictions_with_boxes, confidence_threshold, iou_threshold=0.4):

        predictions_part1 = predictions_with_boxes[:, :, 0:5]
        conf_mask = predictions_with_boxes[:, :, 5:] > confidence_threshold
        predictions_part2 = predictions_with_boxes[:, :, 5:] * conf_mask
        predictions = tf.concat([predictions_part1,predictions_part2],-1)
        predictions = predictions.numpy()
        result = {}
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            print("shape1", shape)
            non_zero_idxs = np.nonzero(image_pred)

            idx = list(set(non_zero_idxs[0]))
            # idx = non_zero_idxs[0]
            image_pred = image_pred[idx]
            print("shape2", image_pred.shape)
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis=-1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                cls_scores = cls_boxes[:, -1]
                cls_boxes = cls_boxes[:, :-1]

                while len(cls_boxes) > 0:
                    box = cls_boxes[0]
                    score = cls_scores[0]
                    if not cls in result:
                        result[cls] = []
                    result[cls].append((box, score))
                    cls_boxes = cls_boxes[1:]
                    ious = np.array([self.iou(box, x) for x in cls_boxes])
                    iou_mask = ious < iou_threshold
                    cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]

        return result

    def convert_to_original_size(self, box, size, original_size):
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
        return list(box.reshape(-1))

    # 将级别结果显示在图片上
    def draw_boxes(self, i, boxes, img_file, cls_names, detection_size):
        img = Image.open(img_file)
        draw = ImageDraw.Draw(img)

        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))
            for box, score in bboxs:
                box = self.convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                draw.rectangle(box, outline=color)
                draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
                print('{} {:.2f}%'.format(cls_names[cls], score * 100), box[:2])
        img.save(f"output_img{i}.jpg")
        img.show()
        pass
    pass

if __name__ == "__main__":
    input = np.random.randn(2,300,300,3)
    input = tf.constant(input,dtype=tf.float32)
    x = tf.constant(input,dtype=tf.float32)
    print(x.shape)
    print(x.get_shape().as_list()[1:-1])
    model = SSD300Net()
    model.detect(input)
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
    # print(f"loc_preds:{loc_preds[0].shape}")
    # print(f"loc_preds:{loc_preds[1].shape}")
    # print(f"loc_preds:{loc_preds[2].shape}")
    # print(f"loc_preds:{loc_preds[3].shape}")
    # print(f"loc_preds:{loc_preds[4].shape}")
    # print(f"loc_preds:{loc_preds[5].shape}")
    # print(f"cls_preds:{cls_preds[0].shape}")
    # print(f"cls_preds:{cls_preds[1].shape}")
    # print(f"cls_preds:{cls_preds[2].shape}")
    # print(f"cls_preds:{cls_preds[3].shape}")
    # print(f"cls_preds:{cls_preds[4].shape}")
    # print(f"cls_preds:{cls_preds[5].shape}")
    # tem = tf.concat([loc_preds[0], cls_preds[0]],-1)
    # tem = tf.reshape(tem,(tem.shape[0], -1, tem.shape[-1]))
    # print(tem.shape)
    # x = range(0, 13)
    # y = range(0, 13)
    # a, b = tf.meshgrid(x, y)

    pass
