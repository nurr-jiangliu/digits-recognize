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

1. 其次，您还需要安装 opencv sklearn scipy numpy 

1. 更重要的是，你需要 python3.6

1. 最后 Enjoy it

    ```
    git clone https://github.com/JunJun-Love-Amber/digits-recognize-by-svc-service.git 
    cd digits-recognize-by-svc-service && sudo python3 main.py
    ```

# 接口

## 1.获取结果 GET： ```/get/result/<uuid>```

**uuid 是调用上传图片后返回的标志**

##### 返回数据：

```json
{
    "code": 0, # 继续轮询
    "message":"识别中"
}
{
    "code": 200, # 停止轮询
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

var options = { method: 'POST',
  url: 'http://127.0.0.1:80/savePicture',
  headers: 
   { 'postman-token': '173509cf-c706-b6f2-8cad-8e048743b3df',
     'cache-control': 'no-cache',
     'content-type': 'application/json' },
  body: { picture: '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' },
  json: true };

request(options, function (error, response, body) {
  if (error) throw new Error(error);

  console.log(body);
});

```
