# digits-recognize-by-svc-service
入门级实验，使用sklearn做的手写数字识别服务


# helper


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
    sudo git clone https://github.com/JunJun-Love-Amber/digits-recognize-by-svc-service.git 
    cd digits-recognize-by-svc-service && sudo python3 main.py
    ```

