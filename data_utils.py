#-*- coding:utf-8 _*-

import pandas as pd
import numpy as np
from utils.cut_words import *

headers = ["id", "q1", "q2", "label"]
data1 = pd.read_csv("data/atec_nlp_sim_train.csv", header=None, names=headers, sep="\t")
data2 = pd.read_csv("data/atec_nlp_sim_train_add.csv", header=None, names=headers, sep="\t")
print(data1.shape)
print(data1.head())
print(data2.shape)
print(data2.head())

##
stopwords_path = "dict/stop_words.txt"
stopwords_set = set([value.strip().replace("\n", "") for value in open(stopwords_path, encoding="utf8").readlines()])

dataset = pd.concat([data1, data2], ignore_index=False)
dataset["text1"] = dataset["q1"].str.strip().str.lower()
dataset["text2"] = dataset["q2"].str.strip().str.lower()
dataset["cut1"] = dataset.apply(lambda row: seg_cut(row["text1"], stopwords_set), axis=1)
dataset["cut2"] = dataset.apply(lambda row: seg_cut(row["text2"], stopwords_set), axis=1)
dataset["cut_clean1"] = dataset["cut1"].map(cleantext)
dataset["cut_clean2"] = dataset["cut2"].map(cleantext)
print("分词结束")

##划分训练集、验证集和测试集
ratio = (0.8, 0.1, 0.1)
samples = dataset.shape[0]
num_train = int(samples * ratio[0])
num_val = int(samples * ratio[1])

# split the data into training set, validation set and test set
indices = np.arange(samples)
np.random.shuffle(indices)

headers_1 = ["cut_clean1", "cut_clean2", "label"]
features = dataset[headers_1].values
features = features[indices]
features_train, features_val, features_test = features[:num_train], features[num_train:num_train + num_val], \
features[num_train + num_val:]

pd.DataFrame(features_train, columns=headers_1).to_csv("ss/ss_train.tsv", sep="\t")
pd.DataFrame(features_val, columns=headers_1).to_csv("ss/ss_dev.tsv", sep="\t")
pd.DataFrame(features_test, columns=headers_1).to_csv("ss/ss_test.tsv", sep="\t")
