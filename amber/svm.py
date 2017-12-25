import base64 as base64lib
import json as jsonlib
import os
import shutil
import sys

import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from redis import Redis
import numpy as np
import cv2


class SVM:
    def __init__(self, redis: Redis):
        self.clf = SVC()
        self.redis = redis
        # if os.path.isdir("./models"):
        #     shutil.rmtree("./models")
        # os.mkdir("./models")
        self.train()
        self.test()
        # self.predict()

    def train(self, dir="./train"):
        print("训练模型...")
        files = os.listdir(dir)
        counts = len(files)

        X = np.zeros((len(files), 1024))
        for index, file in enumerate(files):
            X[index] = self.txt2vector(dir + "/" + file)

        lables = np.zeros(shape=counts, dtype=int)
        for index, file in enumerate(files):
            lables[index] = int(file[0:1])
        self.clf.fit(X, lables)
        # joblib.dump(self.clf, "./models/svm.model")

    def test(self, dir="./train"):
        print("测试模型...")
        # 装载训练好的模型
        # self.clf = joblib.load("./models/svm.model")
        files = os.listdir(dir)
        succeed = 0  #成功数量
        for file in files:
            num = int(file[0:1])
            # print(dir + "/" + file, end='\t\t')
            vector = self.txt2vector(dir + "/" + file)
            if (self.clf.predict(vector) == num):
                succeed += 1
        print("成功率：{}".format(succeed / len(files)))

    # For Redis
    def predict(self):
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
                picture_cv2_origin, (32, 32), interpolation=cv2.INTER_AREA)
            picture_reshape = picture_resize.reshape(1, 1024)
            for i in range(1024):
                picture_reshape[0][i] = int(
                    picture_reshape[0, i] == 255 and "0" or "1")
            predict_num = self.clf.predict(picture_reshape)
            self.redis.set(
                uuid,
                jsonlib.dumps({
                    "code":
                    200,
                    "message":
                    "识别结果：{}（数据集小，不对不要打我/(ㄒoㄒ)/~~）".format(predict_num)
                }))

    def txt2vector(self, filename):
        returnVect = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect

    # For Files

    # def train(dir="./"):
    #     files = os.listdir(dir)
    #     counts = len(files)

    #     X = np.zeros((len(files), 1024))
    #     for index, file in enumerate(files):
    #         X[index] = txt2vector(dir + "/" + file)

    #     for i in range(10):
    #         clf = SVC()
    #         lables = np.zeros(shape=counts, dtype=int)
    #         for index, file in enumerate(files):
    #             num = int(file[0:1])
    #             lables[index] = (num == i) and 1 or -1
    #         clf.fit(X, lables)
    #         joblib.dump(clf, "./models/{}.model".format(i))

    # def predict(dir="./"):
    #     # 装载训练好的模型
    #     cls = joblib.load("./models/svm.model")
    #     files = os.listdir(dir)
    #     succeed = 0  #成功数量
    #     for file in files:
    #         num = int(file[0:1])
    #         print(dir + "/" + file, end='\t\t')
    #         vector = img2vector(dir + "/" + file)

    #         predict_num = cls.predict(vector)
    #         if (predict_num == num):
    #             succeed += 1
    #             print(str(predict_num))
    #         else:
    #             print("识别失败为：{}".format(predict_num))
    #     print("成功率：{}".format(succeed / len(files)))

    # def img2vector(filename):
    #     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #     out = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    #     out = out.reshape(1, 1024)
    #     for i in range(1024):
    #         out[0][i] = int(out[0, i] == 255 and "0" or "1")
    #     return out
