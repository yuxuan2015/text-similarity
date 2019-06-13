#-*- coding:utf-8 _*-


import os
from module import SS


ss_model = SS()

##lightnlp文本相似性计算
train_path = 'ss/ss_train.tsv'
dev_path = 'ss/ss_dev.tsv'
test_path = 'ss/ss_test.tsv'
vec_path = ''


##模型保存路径
model_path = "./ss_saves"
print(model_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

ss_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path=model_path)

#ss_model.load(model_path)
ss_model.test(test_path)
