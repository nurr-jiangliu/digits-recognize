#coding=utf-8
import base64 as base64lib
import json as jsonlib
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from redis import Redis
from tensorflow.examples.tutorials.mnist import input_data

import cv2


# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True), b), name=name)


# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


class AlexNet:
    def __init__(self):
        self.initOK = False
        self.redis = redis

    def train(self, training_iters=200000):
        mnist = input_data.read_data_sets("./temp", one_hot=True)
        # 定义网络超参数
        batch_size = 64
        dropout = 0.75
        # 占位符输入
        tf_x, tf_y, tf_dropout, _, _ = self.init()
        # 构建模型
        predict = self.alexnet()
        # 定义损失函数和学习步骤
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=tf_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(tf_y, 1)), tf.float32))
        tf_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=tf_y))

        # 开启一个训练
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 获取批数据
                session.run(optimizer, feed_dict={tf_x: batch_xs, tf_y: batch_ys, tf_dropout: dropout})
                if step % 20 == 0 or True:
                    # 计算精度
                    loss = session.run(tf_cost, feed_dict={tf_x: mnist.test.images[:256], tf_y: mnist.test.labels[:256], tf_dropout: 1.})
                    # 计算损失值
                    accuracy = session.run(tf_accuracy, feed_dict={tf_x: mnist.test.images[:256], tf_y: mnist.test.labels[:256], tf_dropout: 1.})
                    # accuracy, loss = self.calc_accuracy(session, mnist.test.images[:256], mnist.test.labels[:256])
                    print("Accuracy =  " + "{:.5f}".format(accuracy) + "  Loss = " + "{:.6f}".format(loss))
                step += 1
            saver = tf.train.Saver(dict(self.weights,**self.biases))
            saver.save(session, './models/alexnet/')
            print("Optimization Finished!")

    def test(self):
        mnist = input_data.read_data_sets("./temp", one_hot=True)
        # 占位符输入
        tf_x, tf_y, tf_dropout, _, _ = self.init()
        # 构建模型
        predict = self.alexnet()
        tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(tf_y, 1)), tf.float32))
        tf_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=tf_y))

        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, "./models/alexnet/")
            # 计算精度
            loss = session.run(tf_cost, feed_dict={tf_x: mnist.test.images[:256], tf_y: mnist.test.labels[:256], tf_dropout: 1.})
            # 计算损失值
            accuracy = session.run(tf_accuracy, feed_dict={tf_x: mnist.test.images[:256], tf_y: mnist.test.labels[:256], tf_dropout: 1.})
            print("Accuracy =  " + "{:.5f}".format(accuracy) + "  Loss = " + "{:.6f}".format(loss))

    # For Redis
    def predict(self):

        # 占位符输入
        tf_x, tf_y, tf_dropout, _, _ = self.init()
        # 恢复模型 so easy
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./models/alexnet/")
            while True:
                json = self.redis.brpop('pictures')
                data = jsonlib.loads(json[1])
                uuid = data['uuid']
                base64 = data['picture']
                picture = base64lib.b64decode(base64)
                picture_mat = np.fromstring(picture, np.uint8)
                picture_cv2_origin = cv2.imdecode(picture_mat, cv2.IMREAD_GRAYSCALE)
                picture_resize = cv2.resize(picture_cv2_origin, (28, 28), interpolation=cv2.INTER_AREA)

                picture_resize = picture_resize.astype(float)
                picture_reshape = picture_resize.reshape(1, 28 * 28)
                for i in range(28 * 28):
                    picture_reshape[0][i] = (255 - picture_reshape[0][i]) / 255

                for row in range(28):
                    for col in range(28):
                        if 0.0 == picture_reshape[0][row * 28 + col]:
                            print("0", end='')
                        else:
                            print("1", end='')
                    print('')
                print('')

                predict_num = sess.run(tf.argmax(tf_y, 1), feed_dict={tf_x: picture_reshape})
                json = jsonlib.dumps({"code": 200, "message": "识别结果：{}（数据集小，不对不要打我/(ㄒoㄒ)/~~）".format(predict_num[0])})
                self.redis.set(uuid, json)

    def init(self):
        if self.initOK == False:
            # 存储所有的网络参数
            self.weights = {
                'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64]), name='wc1'),
                'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192]), name='wc2'),
                'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384]), name='wc3'),
                'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384]), name='wc4'),
                'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256]), name='wc5'),
                'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 4096]), name='wd1'),
                'wd2': tf.Variable(tf.random_normal([4096, 4096]), name='wd2'),
                'wout': tf.Variable(tf.random_normal([4096, 10]), name='wout')
            }
            self.biases = {
                'bc1': tf.Variable(tf.random_normal([64]), name='bc1'),
                'bc2': tf.Variable(tf.random_normal([192]), name='bc2'),
                'bc3': tf.Variable(tf.random_normal([384]), name='bc3'),
                'bc4': tf.Variable(tf.random_normal([384]), name='bc4'),
                'bc5': tf.Variable(tf.random_normal([256]), name='bc5'),
                'bd1': tf.Variable(tf.random_normal([4096]), name='bd1'),
                'bd2': tf.Variable(tf.random_normal([4096]), name='bd2'),
                'bout': tf.Variable(tf.random_normal([10]), name='bout')
            }
            self.tf_x = tf.placeholder(tf.float32, [None, 784], name='x')
            self.tf_y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.tf_dropout = tf.placeholder(tf.float32, name="dropout")
            self.initOK = True
        return self.tf_x, self.tf_y, self.tf_dropout, self.weights, self.biases

    def alexnet(self):
        x, y, dropout, weights, biases = self.init()
        # 向量转为矩阵
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 第一层卷积
        # 卷积
        conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
        # 下采样
        pool1 = max_pool('pool1', conv1, k=2)
        # 归一化
        norm1 = norm('norm1', pool1, lsize=4)

        # 第二层卷积
        # 卷积
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        # 下采样
        pool2 = max_pool('pool2', conv2, k=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=4)

        # 第三层卷积
        # 卷积
        conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        # 归一化
        norm3 = norm('norm3', conv3, lsize=4)

        # 第四层卷积
        # 卷积
        conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
        # 归一化
        norm4 = norm('norm4', conv4, lsize=4)

        # 第五层卷积
        # 卷积
        conv5 = conv2d('conv5', norm4, weights['wc5'], biases['bc5'])
        # 下采样
        pool5 = max_pool('pool5', conv5, k=2)
        # 归一化
        norm5 = norm('norm5', pool5, lsize=4)

        # 全连接层1，先把特征图转为向量
        dense1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1')
        dense1 = tf.nn.dropout(dense1, dropout)

        # 全连接层2
        dense2 = tf.reshape(dense1, [-1, weights['wd2'].get_shape().as_list()[0]])
        dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')  # Relu activation
        dense2 = tf.nn.dropout(dense2, dropout)

        # 网络输出层
        out = tf.matmul(dense2, weights['out']) + biases['out']
        return out


# # 开启一个训练
# with tf.Session() as session:
#     saver = tf.train.Saver()
#     saver.restore(session, "./models/alexnet/")
#     print(session.run(tf.argmax(predict, 1), feed_dict={x: mnist.test.images[:1], keep_prob: 1.}))
#     print(mnist.test.labels[0])
