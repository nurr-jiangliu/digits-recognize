# digits-recognize-by-svc-service
入门级实验，使用sklearn做的手写数字识别服务


# 预备
 

1.  首先，你需要安装 [Redis](https://redis.io/) 作为KV数据库

    ```
    sudo mkdir -p /usr/local/src
    cd /usr/local/src
    sudo wget http://download.redis.io/releases/redis-4.0.6.tar.gz
    sudo tar -xf redis-4.0.6.tar.gz
    cd  redis-4.0.6
    sudo make 
    sudo make install
    
    ```

1. 安装好了后，你需要执行 ``` redis-server --port 6379 &```    启动 redis，若需要其他IP/端口，请修改 ```main.py``` 相应代码

1. 其次，您还需要安装 opencv-python sklearn scipy numpy  pillow redis flask flask_cors

1. 更重要的是，你需要 python3.6

1. 最后 Enjoy it

    ```
    git clone https://github.com/JunJun-Love-Amber/digits-recognize-by-svc-service.git 
    cd digits-recognize-by-svc-service && sudo python3 main.py
    ```

# 接口

## 1.获取结果 GET： ```/getResult/<uuid>```

**uuid 是调用上传图片后返回的标志**

##### 返回数据：

```j
{
    "code": 0,   # 继续轮询
    "message":"识别中"
}
{
    "code": 200,  # 停止轮询
    "message": "识别结果：[0]（数据集小，不对不要打我/(ㄒoㄒ)/~~）"
}
```
### 2.上传图片 POST： ```/savePicture```

###### 返回例子：
```json
{
    "uuid": "xxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```
###### 请求例子：

```
var request = require("request");

var options = { 
    method: 'POST',
    url: 'http://127.0.0.1:80/savePicture',
    headers:  { 
        'cache-control': 'no-cache',
        'content-type': 'application/json' 
    },
    body: { picture: 'base64_encode(picture_data)' },
    json: true 
};

request(options, function (error, response, body) {
    if (error) throw new Error(error);
    console.log(body);
});

```
