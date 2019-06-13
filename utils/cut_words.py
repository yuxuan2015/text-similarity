# -*- coding:utf-8 -*-

from pyhanlp import *
import numpy as np

HanLP.Config.ShowTermNature = False
###不保留词性(分去停用词和不去停用词)
#分词不去停用词
def seg_cut(text, stopwords_set):
    return ' '.join([str(term) for term in HanLP.segment(text.strip()) if str(term) not in stopwords_set])


##清洗字符串
import re, sys, unicodedata


res = re.compile(r'\s+')
red = re.compile(r'^(\d+)$')
punc = re.compile(r'[\.\!\/,$%^*)(\+\<\[\]\"\']+|[——！，。·？?_、~@#￥%……&*（）：:]+')
# 清洗标点符号等异常字符
todel = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)) not in ('Lu', 'Ll', 'Lt', 'Lo', 'Nd', 'Nl', 'Zs'))

# 清洗分词结果的方法
# 清洗分词结果的方法
def cleantext(text):
    if text != '' and text is not np.nan:
        # return re.sub(res, ' ', ' '.join(map(lambda x: re.sub(red, '', x), text.translate(todel).split(' ')))).strip()
        return re.sub(res, ' ', ' '.join(map(lambda x: re.sub(red, '', x), re.sub(punc, '', text).split(' ')))).strip()
    else:
        return text


def cleanchar(text):
    if text != '' and text is not np.nan:
        return re.sub(res, ' ', ' '.join(map(lambda x: re.sub(red, '', x), text.translate(todel).split(' ')))).strip()
    else:
        return text
