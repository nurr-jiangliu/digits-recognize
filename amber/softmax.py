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


class SoftMax:
    def __init__(self, redis: Redis):
        self.initOK = False
        self.redis = redis
        # if os.path.isdir("./models"):
        #     shutil.rmtree("./models")
        # os.mkdir("./models")
        # self.train()
        # self.test()

    def train(self):
        print("训练模型...")
        # Create the model
        mnist = input_data.read_data_sets("./temp")
        x, W, b, y, y_ = self.init()
        # Define loss and optimizer
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
            cross_entropy)
        # Train
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for _ in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            saver = tf.train.Saver()
            saver.save(sess, './models/softmax/')

    def test(self, dir="./train"):
        print("测试模型...")
        # Create the model
        x, W, b, y, y_ = self.init()
        mnist = input_data.read_data_sets("./temp")
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 恢复模型 so easy
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./models/softmax/")
            success_rate = sess.run(
                accuracy,
                feed_dict={
                    x: mnist.test.images,
                    y_: mnist.test.labels
                })
            print("成功率：{}".format(success_rate))

    # For Redis
    def predict(self):

        x, W, b, y, y_ = self.init()

        # 恢复模型 so easy
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./models/softmax/")
            while True:
                json = self.redis.brpop('pictures')
                data = jsonlib.loads(json[1])
                uuid = data['uuid']
                base64 = data['picture']
                picture = base64lib.b64decode(base64)
                picture_mat = np.fromstring(picture, np.uint8)
                picture_cv2_origin = cv2.imdecode(picture_mat,
                                                  cv2.IMREAD_GRAYSCALE)
                picture_resize = cv2.resize(
                    picture_cv2_origin, (28, 28), interpolation=cv2.INTER_AREA)

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

                predict_num = sess.run(
                    tf.argmax(y, 1), feed_dict={
                        x: [picture_reshape[0]]
                    })
                json = jsonlib.dumps({
                    "code":
                    200,
                    "message":
                    "识别结果：{}（数据集小，不对不要打我/(ㄒoㄒ)/~~）".format(predict_num[0])
                })
                self.redis.set(uuid, json)

    def init(self):

        if self.initOK == False:
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            self.W = tf.Variable(tf.zeros([784, 10]), name='W')
            self.b = tf.Variable(tf.zeros([10]), name='b')
            self.y = tf.matmul(self.x, self.W) + self.b
            self.y_ = tf.placeholder(tf.int64, [None], name='y_')
        return self.x, self.W, self.b, self.y, self.y_
