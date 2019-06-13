#-*- coding:utf-8 _*-
"""
@author:lyy
@file: text_similarity.py
@time: 2019/06/12
@Software: PyCharm
"""

from module import SS
import pandas as pd
import os

##加载客服问题相似度计算模型
ss_model = SS()

text = "cut_clean"
main_path = os.path.dirname(os.path.abspath(__file__))
ss_model_path = os.path.join(main_path, 'ss_saves', text)
ss_model.load(ss_model_path)



def text_similar(text):
    text1, text2 = text.split("##")
    prob = float(ss_model.predict(text1, text2))
    print(prob)
    return prob


def text_similar_multiple(texts):
    df = pd.DataFrame([value.split("##") for value in texts], columns=["q1", "q2"])
    df["prob"] = pd.Series(ss_model.predict_multiple(df["q1"].values, df["q2"].values))
    df = df.sort_values(by="prob", ascending=False)
    return df

if __name__ == "__main__":
    texts = ["怎么更改花呗手机号码##我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号", "也开不了花呗，就这样了？完事了##真的嘛？就是花呗付款", "花呗冻结以后还能开通吗##我的条件可以开通花呗借款吗", "如何得知关闭借呗##想永久关闭借呗", "花呗扫码付钱##二维码扫描可以用花呗吗", "花呗逾期后不能分期吗##我这个 逾期后还完了 最低还款 后 能分期吗", "花呗分期清空##花呗分期查询", "借呗逾期短信通知##如何购买花呗短信通知", "借呗即将到期要还的账单还能分期吗##借呗要分期还，是吗"]
    df = text_similar_multiple(texts)
    print(df)
    print(text_similar(texts[0]))
