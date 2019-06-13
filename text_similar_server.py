#-*- coding:utf-8 _*-

from module import SS
from sanic import Sanic
from sanic import response
import json
import os
import pandas as pd

app = Sanic()

#bind = "0.0.0.0:8888"
ss_model = SS()

text = "cut_clean"
ss_model_path = os.path.join('ss_saves', text)
ss_model.load(ss_model_path)



def ss_predict(text):
    text1, text2 = text.split("#")
    return float(ss_model.predict(text1, text2))

##同时处理多条
def ss_predict_multiple(texts):
    probs = list(map(ss_predict, texts))
    return probs


@app.route('/kefu_text_similar', methods=['POST'])
async def text_similar(request):
    request_json = request.body
    print(request_json)
    input_json = json.loads(request_json.decode('utf8'))
    texts = input_json["texts"]
    print(texts)
    text1, text2 = texts.split("##")
    prob = float(ss_model.predict(text1, text2))
    print(prob)
    return response.json({"result": prob})


@app.route('/text_similar', methods=['POST'])
async def text_similar(request):
    request_json = request.body
    print(request_json)
    input_json = json.loads(request_json.decode('utf8'))
    texts = input_json["texts"]
    print(texts)
    df = pd.DataFrame([value.split("##") for value in texts], columns=["q1", "q2"])
    probs = ss_model.predict_multiple(df["q1"].values, df["q2"].values)
    # probs = list(map(ss_predict, texts))
    return response.json({"result": probs})


if __name__ == "__main__":
    # app.run(host="192.168.31.232", port=8600)
    app.run(host="localhost", port=8600)
