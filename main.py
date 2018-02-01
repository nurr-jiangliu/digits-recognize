#encoding:utf-8
import base64 as base64lib
import hashlib
import json as jsonlib
import os
import sys
from io import BytesIO
from threading import Thread

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from PIL import Image
from redis import Redis

import amber

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    
    return 'Hello World!'


@app.route('/savePicture', methods=['POST'])
def savePicture():
    # 获取数据
    data = request.json
    # 计算图片哈希值
    hashsss = hashlib.md5()
    hashsss.update(data['picture'].encode("utf8"))
    # 展示图片
    # im = Image.open(BytesIO(base64lib.b64decode(data['picture'])))
    # im.show()

    uuid = hashsss.hexdigest()

    # 设置图片唯一标识，以便返回前端轮询
    data['uuid'] = uuid

    # 保存 Redis
    redis.lpush('pictures', jsonlib.dumps(data))
    redis.set(uuid, jsonlib.dumps({"code": 0, "message": "识别中，请耐心等待"}))
    return jsonify({'uuid': uuid})


@app.route('/getResult/<uuid>', methods=['GET'])
def getResult(uuid):
    data = jsonlib.loads(redis.get(uuid))
    if data['code'] == 200:        
        # redis.delete(uuid)
        pass
    return jsonify(data)


if __name__ == '__main__':
    redis = Redis(host='127.0.0.1', port=6379, db=0)
    print("redis:" + str(redis.ping()))
    # svm = amber.svm.SVM(redis)
    # Thread(target=svm.predict).start()
    softmax = amber.softmax.SoftMax(redis)
    Thread(target=softmax.predict).start()
    app.run(host="0.0.0.0", port=8888, debug=False, threaded=True)
