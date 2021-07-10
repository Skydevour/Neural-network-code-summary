#-*- coding:utf-8 -*- 
# @Time : 2021/1/10 10:41 
# @Author : 万志杨
# @File : SEnet_fashionMNIST.py 
# @Software: PyCharm
import pandas as pd
import numpy as np
import warnings
import os
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import fashion_MNIST_load

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 获取数据
def get_data():
    mnist = np.load("MNIST_data/", one_hot=True)
    return mnist

# 设置权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 设置b
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 设置卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def SE_block(x,ratio):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    print(shape)
    with tf.VariableSco("squeeze_and_excitation"):
        # 第一层 全局平均池化层
        squeeze = tf.nn.avg_pool(x,[1,shape[1],shape[2],1],[1,shape[1],shape[2],1],padding="SAME")
        # 第二层 全连接层
        w_excitation1 = weight_variable(([1,1,channel_out,channel_out/ratio]))
        b_excitation1 = bias_variable([channel_out/ratio])
        excitation1 = conv2d(squeeze,w_excitation1) + b_excitation1
        excitation1_output = tf.nn.relu(excitation1)
        # 第三层 全连接层
        w_excitation2 = weight_variable([1,1,channel_out/ratio,channel_out])
        b_excitation2 = bias_variable([channel_out])
        excitation2 = conv2d(excitation1_output,w_excitation2) + b_excitation2
        excitation2_output = tf.nn.sigmoid(excitation2)
        # 第四层
        excitation_output = tf.reshape(excitation2_output,[-1,1,1,channel_out])
        h_output = excitation_output * x
    return h_output

if __name__ == "__main__":
    # 输入
    x = tf.placeholder(tf.float32,[None,784])
    # 转换成二维图片
    x_image = tf.reshape(x,[-1,28,28,1])
    # 第一层卷积层
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h1_conv = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h1_conv = SE_block(h1_conv,4)
    # 配置池化层
    h1_pool = pool(h1_conv)
    # 第二层卷积层
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h2_conv = tf.nn.relu(conv2d(h1_pool,W_conv2) + b_conv2)
    h2_conv = SE_block(h2_conv,4)
    # 第二层池化层
    h2_pool = pool(h2_conv)

    # 全连接层
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h2_pool_flat = tf.reshape(h2_pool,shape=[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h2_pool_flat,W_fc1) + b_fc1)

    # 配置dropout层
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    # 配置全连接2层
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    y_label = tf.placeholder(tf.float32,[None,10])
    mnist = get_data()

    # 交叉熵
    cross = -tf.reduce_sum(y_label * tf.log(y_predict))
    # 梯度下降法
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross)
    # 求准确率
    coo = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_label,1))
    acc = tf.reduce_mean(tf.cast(coo,"float"))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(200):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            train_acc = acc.eval(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_acc))
            train_step.run(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 0.5})
    point = 0
    point_temp = 0
    for i in range(10):
        testSet = mnist.test.next_batch(50)
        point_temp = acc.eval(feed_dict={x: testSet[0], y_label: testSet[1], keep_prob: 1.0})
        print("test accuracy %g" % point_temp)
        point += point_temp
    print("test accuracy:" + str(point / 10))


