import amber
from redis import Redis
amber.svm.SVM(Redis(host='127.0.0.1', port=6379, db=0))
