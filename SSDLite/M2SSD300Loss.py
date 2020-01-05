#-- coding: utf-8 --
import tensorflow as tf
import numpy as np

class SSDLoss():
    def __init__(self,neg_pos_ratio = 3,n_neg_min = 0,alpha=1.0):
        self.neg_pos_ratio = neg_pos_ratio #负例和正例的比值
        self.n_neg_min = n_neg_min
        self.alpha = alpha
        pass

    def smooth_L1_loss(self,y_true,y_pred):
        #y_true.shape和y_pred.shape均为(batch_size, n_boxes, 4)
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        loss = tf.where(tf.less(absolute_loss,1.),square_loss,absolute_loss - 0.5)
        return tf.reduce_sum(loss,axis=-1)
        pass

    def log_loss(self,y_true,y_pred):
        #y_true.shape和y_pred.shape均为(batch_size, n_boxes, n_classes)
        y_pred = tf.maximum(y_pred,1e-15)#确保y_pred不为0
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss
        pass

    def compute_loss(self,y_true,y_pred):
        #y_true.shape和y_pred.shape均为(batch_size, n_boxes, n_classes+4+8)
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]

        # 1: Compute the losses for class and box predictions for every box.
        cls_loss = self.log_loss(y_true[:,:,:-12],y_pred[:,:,:-12])
        loc_loss = self.smooth_L1_loss(y_true[:,:-12:-8],y_pred[:,:-12:-8])

        # 2: Compute the classification losses for the positive and negative targets

        # Create masks for the positive and negative ground truth classes.
        negatives = y_true[:,:,0] #背景
        positives = tf.reduce_max(y_true[:,:,1:-12],axis=-1)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)
        pos_cls_loss = tf.reduce_sum(cls_loss * positives,axis=-1)

        # Compute the classification loss for the negative default boxes (if there are any).
        neg_cls_loss_all = cls_loss * negatives
        n_neg_losses = tf.count_nonzero(neg_cls_loss_all,dtype=tf.int32)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * n_positive,self.n_neg_min),n_neg_losses)

        # In the unlikely case when either there are no negative ground truth boxes at all
        # or the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
            pass

        #Otherwise compute the negative loss.
        def f2():
            neg_cls_loss_all_1D = tf.reshape(neg_cls_loss_all,[-1])
            values,indices = tf.nn.top_k(neg_cls_loss_all_1D,k=n_negative_keep,sorted=False)

            indices = tf.expand_dims(indices,axis=1)
            updates = tf.ones_like(indices,dtype=tf.int32)
            shape = tf.shape(neg_cls_loss_all_1D)# Tensor of shape (batch_size * n_boxes,)

            negtives_keep = tf.scatter_nd(indices,updates,shape)
            negtives_keep = tf.to_float(tf.shape(negtives_keep,[batch_size,n_boxes]))
            neg_cls_loss = tf.reduce_sum(cls_loss * negtives_keep,axis=-1)
            return neg_cls_loss
            pass
        neg_cls_loss = tf.cond(tf.equal(n_neg_losses,0),f1,f2)
        cls_loss = pos_cls_loss + neg_cls_loss

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes
        #    (obviously: there are no ground truth boxes they would correspond to).
        loc_loss = tf.reduce_sum(loc_loss * positives,axis=-1)

        # 4: Compute the total loss.
        total_loss = (cls_loss + self.alpha * loc_loss) / tf.maximum(1.0,n_positive)
        total_loss = total_loss * batch_size
        return total_loss
        pass

    pass