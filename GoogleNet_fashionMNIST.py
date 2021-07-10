#-*- coding:utf-8 -*- 
# @Time : 2021/1/6 15:14 
# @Author : 万志杨
# @File : GoogleNet_fashionMNIST.py 
# @Software: PyCharm

import pandas as pd
import numpy as np
import warnings
import os
import math
import random
import tensorflow as tf
import fashion_MNIST_load
from tensorflow import keras
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class ConvBNRelu(keras.Model):

    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x

    # Inception Block 模块。


class InceptionBlk(keras.Model):

    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()

        self.ch = ch
        self.strides = strides

        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernelsz=3, strides=1)

        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        # concat along axis=channel    通道数
        x = tf.concat([x1, x2, x3_2, x4], axis=3)

        return x


# Res Block 模块。继承keras.Model或者keras.Layer都可以
class Inception(keras.Model):

    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch

        self.conv1 = ConvBNRelu(init_ch)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        for block_id in range(num_layers):

            for layer_id in range(2):

                if layer_id == 0:

                    block = InceptionBlk(self.out_channels, strides=2)

                else:
                    block = InceptionBlk(self.out_channels, strides=1)

                self.blocks.add(block)

            # enlarger out_channels per block
            self.out_channels *= 2

        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, x, training=None):

        out = self.conv1(x, training=training)

        out = self.blocks(out, training=training)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out

def preprocess(x, y):  # 数据预处理
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batchsize = 512

(x_train, y_train), (x_test, y_test) = fashion_MNIST_load.load_data()
print(x_train.shape, y_train.shape)

# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

# 训练集预处理
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 构造数据集,这里可以自动的转换为tensor类型了
db_train = db_train.map(preprocess).shuffle(10000).batch(batchsize)

# 测试集预处理
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构造数据集
db_test = db_test.map(preprocess).shuffle(10000).batch(batchsize)

db_iter = iter(db_train)
sample = next(db_iter)
print("batch: ", sample[0].shape, sample[1].shape)

# 第一个为残差块数，第二个为类别数
model = Inception(2,10)
model.build(input_shape=(None,28,28,1))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 分类器
c = keras.losses.CategoricalCrossentropy(from_logits=True)
acc_meter = keras.metrics.Accuracy()

for epoch in range(10):
    for step, (x,y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = c(tf.one_hot(y,depth=10),logits)

        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step % 20 == 0:
            print(epoch, step, 'loss:',loss.numpy())

    # 测试集测试
    acc_meter.reset_states()
    for x, y in db_test:
        # [b, 10]
        logits = model(x, training=False)
        # [b, 10] => [b]
        pred = tf.argmax(logits, axis=1)
        # [b] vs [b, 10]
        acc_meter.update_state(y, pred)
    print(epoch, 'evaluation acc:', acc_meter.result().numpy())


