#-*- coding:utf-8 _*-

from predict import text_similar, text_similar_multiple
from sanic import Sanic
from sanic import response
import json


app = Sanic()


@app.route('/kefu_text_similar', methods=['POST'])
async def text_similar(request):
    request_json = request.body
    print(request_json)
    input_json = json.loads(request_json.decode('utf8'))
    texts = input_json["texts"]
    print(texts)
    prob = text_similar(texts)
    print(prob)
    return response.json({"result": prob})


@app.route('/text_similar', methods=['POST'])
async def text_similar(request):
    request_json = request.body
    print(request_json)
    input_json = json.loads(request_json.decode('utf8'))
    texts = input_json["texts"]
    print(texts)
    df = text_similar_multiple(texts)
    return response.json({"result": df["prob"].tolist()})


if __name__ == "__main__":
    app.run(host="localhost", port=8600)
