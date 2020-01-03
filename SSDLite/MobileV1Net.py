#-- coding: utf-8 --

import tensorflow as tf

class MobileV1Net(tf.keras.Model):
    def __init__(self):
        super(MobileV1Net,self).__init__()
        self.conv_stage_0 = Conv_BN_ReLU(32, 3, 2)
        self.conv_stage_1 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(64,1)
        self.conv_stage_2 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(128,2)
        self.conv_stage_3 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(128,1)
        self.conv_stage_4 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(256,2)
        self.conv_stage_5 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(256,1)
        self.conv_stage_6 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,2)
        self.conv_stage_7a = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_7b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_7c = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_7d = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_7e = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_8 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(1024,2)
        self.conv_stage_9 = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(1024,1)
        self.conv_stage_10 = tf.keras.layers.AveragePooling2D(7,1)
        self.conv_stage_11 = tf.keras.layers.Dense(1000)
        self.conv_stage_12 = tf.keras.layers.Softmax()
        pass

    def call(self,input):
        x = self.conv_stage_0(input)  #输入(1, 224, 224, 3)
        x = self.conv_stage_1(x)      #输入(1, 112, 112, 32)
        x = self.conv_stage_2(x)      #输入(1, 112, 112, 64)
        x = self.conv_stage_3(x)      #输入(1, 56, 56, 128)
        x = self.conv_stage_4(x)      #输入(1, 56, 56, 128)
        x = self.conv_stage_5(x)      #输入(1, 28, 28, 256)
        x = self.conv_stage_6(x)      #输入(1, 28, 28, 256)
        x = self.conv_stage_7a(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_7b(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_7c(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_7d(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_7e(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_8(x)      #输入(1, 14, 14, 512)
        x = self.conv_stage_9(x)      #输入(1, 7, 7, 1024)
        x = self.conv_stage_10(x)     #输入(1, 7, 7, 1024)
        x = self.conv_stage_11(x)     #输入(1, 1, 1, 1024)
        x = self.conv_stage_12(x)     #输入(1, 1, 1, 1000)
        return x
        pass
    pass

def DepthwiseConv_BN_ReLU_Conv_BN_ReLU(filters,strides):
    # kernel_size,
    # strides = (1, 1),
    # padding = 'valid',
    # depth_multiplier = 1,
    # data_format = None,
    # activation = None,
    # use_bias = True,
    # depthwise_initializer = 'glorot_uniform',
    # bias_initializer = 'zeros',
    # depthwise_regularizer = None,
    # bias_regularizer = None,
    # activity_regularizer = None,
    # depthwise_constraint = None,
    # bias_constraint = None,
    # ** kwargs
    return tf.keras.Sequential([
        tf.keras.layers.DepthwiseConv2D(3,strides=strides,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters,1,1,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    pass

def Conv_BN_ReLU(filters,size,strides):

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,size,strides,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    pass


if __name__=="__main__":
    import numpy as np
    input = np.random.random((1,224,224,3))
    model = MobileV1Net()
    res = model(input)
    pass