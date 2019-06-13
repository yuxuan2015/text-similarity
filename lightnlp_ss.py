#-*- coding:utf-8 _*-

import pandas as pd
import os
from module import SS


ss_model = SS()

##创建数据目录
text = "cut_clean"
ss_path = os.path.join("ss", text)
print(ss_path)

if not os.path.exists(ss_path):
    os.makedirs(ss_path)


# ##构建训练接、验证机和测试集
# train = pd.read_csv("data/train.tsv", sep="\t")
# dev = pd.read_csv("data/dev.tsv", sep="\t")
# test = pd.read_csv("data/test.tsv", sep="\t")
#
#
# headers = [text+"1", text+"2", "label"]
# ss_train = train[headers]
# ss_dev = dev[headers]
# ss_test = test[headers]
#
# ss_train.to_csv(ss_path+"/ss_train.tsv", header=False, sep="\t")
# ss_dev.to_csv(ss_path+"/ss_dev.tsv", header=False, sep="\t")
# ss_test.to_csv(ss_path+"/ss_test.tsv", header=False, sep="\t")


##lightnlp文本相似性计算
train_path = ss_path+'/ss_train.tsv'
dev_path = ss_path+'/ss_dev.tsv'
test_path = ss_path+'/ss_test.tsv'
vec_path = 'data/model.vec'
#vec_path = 'data/embd_char.vec'

##模型保存路径
model_path = os.path.join("ss_saves", text)
print(model_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

ss_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path=model_path)

#ss_model.load(model_path)
ss_model.test(test_path)
