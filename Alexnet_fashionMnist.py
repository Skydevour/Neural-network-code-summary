# -*- coding:utf-8 -*-
# @Time : 2020/11/29 8:37 
# @Author : 万志杨
# @File : Alexnet_fashionMnist.py 
# @Software: PyCharm
import pickle
from time import *

from tensorflow.python.training import saver


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from tensorflow.examples.tutorials.mnist import input_data
# 导入数据集
fashion_mnist = input_data.read_data_sets('./fashionMNIST_data/',one_hot=True)
# print(fashion_mnist)
# 定义超参数
lr = 0.001
epoch = 200000
batch_size = 128
display_step = 10

# 定义网络参数
input_num = 784
n_classes = 10
dropout = 0.75

# 定义占位符
x = tf.placeholder(tf.float32,[None,input_num])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# 定义卷积
def conv2d(name,x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)

# 池化
def maxpool2d(name,x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# 规范化
def norm(name,l_input,lsize=4):
    return tf.nn.lrn(l_input,lsize,bias = 1.0,alpha=0.01/9.0,beta=0.75,name=name)

weights = {
    'wc1':tf.Variable(tf.random_normal([11, 11, 1, 96])),# 因为mnist数据集的图像channel为1，故卷积核为11*11*1，个数为96
    'wc2':tf.Variable(tf.random_normal([5, 5, 96, 256])),# 因为论文中的卷积在两个GPU上进行计算，笔记本只有一个GPU，
    'wc3':tf.Variable(tf.random_normal([3, 3, 256, 384])),  # 所以每层卷积的数量为论文中的2倍。
    'wc4':tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5':tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1':tf.Variable(tf.random_normal([4*4*256, 4096])),      # 第一个全连接网络的输入为 4096
    'wd2':tf.Variable(tf.random_normal([4096, 1024])),
    'out':tf.Variable(tf.random_normal([1024, 10]))        # 输出为 10 个类别
}

biases = {
    'bc1':tf.Variable(tf.random_normal([96])),
    'bc2':tf.Variable(tf.random_normal([256])),
    'bc3':tf.Variable(tf.random_normal([384])),
    'bc4':tf.Variable(tf.random_normal([384])),
    'bc5':tf.Variable(tf.random_normal([256])),
    'bd1':tf.Variable(tf.random_normal([4096])),
    'bd2':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def alex_net(x,w,b,dropout):
    x = tf.reshape(x,shape=[-1,28,28,1])
    # 第一层
    conv1 = conv2d('conv1',x,weights['wc1'],biases['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1',conv1,k=2)
    # 规范化
    norm1 = norm('norm1',pool1,lsize=4)

    # 第二层
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 下采样
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三层
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 规范化
    norm3 = norm('norm3', conv3, lsize=4)

    # 第四层
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    # 第五层
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 下采样
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 规范化
    norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1
    fc1 = tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2,dropout)

    # 输出
    out = tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    return out

# 构建模型
pred = alex_net(x,weights,biases,keep_prob)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 训练模型和评估模型
# 初始化变量
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    starttime = time()
    while step * batch_size < epoch:
        batch_x, batch_y = fashion_mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter" + str(step * batch_size) + ",Minibatch Loss=" + "{:.6f}".format(
                loss) + ",Training Accuracy=" + "{:.5f}".format(acc))
        step += 1
    print('Optimization Finishion!')
    print("Testing Accuracy:",
    sess.run(accuracy, feed_dict={x: fashion_mnist.test.images[:256], y: fashion_mnist.test.labels[:256], keep_prob: 1.}))
    endtime = time()
    print(endtime - starttime)