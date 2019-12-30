#-- coding: utf-8 --
import tensorflow as tf


class Vgg512Net(tf.keras.Model):
    def __init__(self):
        super(Vgg512Net, self).__init__()
        self.conv_stage_1a = Conv2D_BN_LeakyReLU(64,3)
        self.conv_stage_1b = Conv2D_BN_LeakyReLU(64,3)

        self.conv_stage_2a = tf.keras.layers.MaxPool2D(2,2,padding='same')
        self.conv_stage_2b = Conv2D_BN_LeakyReLU(128, 3)
        self.conv_stage_2c = Conv2D_BN_LeakyReLU(128, 3)

        self.conv_stage_3a = tf.keras.layers.MaxPool2D(2,2,padding='same')
        self.conv_stage_3b = Conv2D_BN_LeakyReLU(256, 3)
        self.conv_stage_3c = Conv2D_BN_LeakyReLU(256, 3)
        self.conv_stage_3d = Conv2D_BN_LeakyReLU(256, 3)

        self.conv_stage_4a = tf.keras.layers.MaxPool2D(2,2,padding='same')
        self.conv_stage_4b = Conv2D_BN_LeakyReLU(512, 3)
        self.conv_stage_4c = Conv2D_BN_LeakyReLU(512, 3)
        self.conv_stage_4d = Conv2D_BN_LeakyReLU(512, 3)
        self.conv_stage_4e = L2_Normalization(512)

        self.conv_stage_5a = tf.keras.layers.MaxPool2D(2,2,padding='same')
        self.conv_stage_5b = Conv2D_BN_LeakyReLU(512, 3)
        self.conv_stage_5c = Conv2D_BN_LeakyReLU(512, 3)
        self.conv_stage_5d = Conv2D_BN_LeakyReLU(512, 3)

        self.conv_stage_6a = tf.keras.layers.MaxPool2D(3, 1,padding='same')
        self.conv_stage_6b = Conv2D_BN_LeakyReLU(1024, 3,dilation_rate=6)

        self.conv_stage_7 = Conv2D_BN_LeakyReLU(1024, 1)

        self.conv_stage_8a = Conv2D_BN_LeakyReLU(256, 1)
        self.conv_stage_8b = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
        self.conv_stage_8c = Conv2D_BN_LeakyReLU(512, 3, 2, padding='valid')

        self.conv_stage_9a = Conv2D_BN_LeakyReLU(128, 1)
        self.conv_stage_9b = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
        self.conv_stage_9c = Conv2D_BN_LeakyReLU(256, 3, 2, padding='valid')

        self.conv_stage_10a = Conv2D_BN_LeakyReLU(128, 1)
        self.conv_stage_10b = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
        self.conv_stage_10c = Conv2D_BN_LeakyReLU(256, 3, 2, padding='valid')

        self.conv_stage_11a = Conv2D_BN_LeakyReLU(128, 1)
        self.conv_stage_11b = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
        self.conv_stage_11c = Conv2D_BN_LeakyReLU(256, 3, 2, padding='valid')

        self.conv_stage_12a = Conv2D_BN_LeakyReLU(128, 1)
        self.conv_stage_12b = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
        self.conv_stage_12c = Conv2D_BN_LeakyReLU(256, 4, padding='valid')

        pass

    def call(self,input_tensor):#input_tensor.shape=(batch_size,512,512,3)
        x = self.conv_stage_1a(input_tensor)
        x = self.conv_stage_1b(x)

        x = self.conv_stage_2a(x)
        x = self.conv_stage_2b(x)
        x = self.conv_stage_2c(x)

        x = self.conv_stage_3a(x)
        x = self.conv_stage_3b(x)
        x = self.conv_stage_3c(x)
        x = self.conv_stage_3d(x)

        x = self.conv_stage_4a(x)
        x = self.conv_stage_4b(x)
        x = self.conv_stage_4c(x)
        x = self.conv_stage_4d(x)  # 输出((batch_size,64,64,3)
        x = self.conv_stage_4e(x)
        stage_4 = x    #(64*64)

        x = self.conv_stage_5a(x)
        x = self.conv_stage_5b(x)
        x = self.conv_stage_5c(x)
        x = self.conv_stage_5d(x)  # 输出(19*19*512)

        x = self.conv_stage_6a(x)
        x = self.conv_stage_6b(x)

        x = self.conv_stage_7(x)
        stage_7 = x  #(32*32)

        x = self.conv_stage_8a(x)
        x = self.conv_stage_8b(x)
        x = self.conv_stage_8c(x)
        stage_8 = x  #(16*16)

        x = self.conv_stage_9a(x)
        x = self.conv_stage_9b(x)
        x = self.conv_stage_9c(x)
        stage_9 = x  #(8*8)

        x = self.conv_stage_10a(x)
        x = self.conv_stage_10b(x)
        x = self.conv_stage_10c(x)
        stage_10 = x  #(4*4)

        x = self.conv_stage_11a(x)
        x = self.conv_stage_11b(x)
        x = self.conv_stage_11c(x)
        stage_11 = x  #(2*2)

        x = self.conv_stage_12a(x)
        x = self.conv_stage_12b(x)
        x = self.conv_stage_12c(x)
        stage_12 = x  # (1*1)

        #[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        return [stage_4, stage_7, stage_8, stage_9, stage_10, stage_11, stage_12]
        pass
    pass


class L2_Normalization(tf.keras.layers.Layer):
    def __init__(self, n_channels):
        super(L2_Normalization, self).__init__()

        value = tf.ones((n_channels,),dtype=tf.dtypes.float32)
        self.gamma = tf.Variable(name="gamma", shape=(n_channels,), dtype=tf.float32, trainable=True,initial_value= value)
        pass

def Conv2D_BN_LeakyReLU(filters, kernel_size, strides=(1, 1),padding='same',dilation_rate=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,dilation_rate=dilation_rate),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    pass

if __name__ == "__main__":
    import numpy as np
    input_tensor = tf.constant(np.random.rand(2,512,512,3),tf.float32)
    model = Vgg512Net()
    stage_4, stage_7, stage_8, stage_9, stage_10, stage_11, stage_12 = model(input_tensor)
    print(stage_4.shape)
    print(stage_7.shape)
    print(stage_8.shape)
    print(stage_9.shape)
    print(stage_10.shape)
    print(stage_11.shape)
    print(stage_12.shape)
    pass