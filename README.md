# 文本相似度计算
 

## 环境依赖
* python3.6
* torch
* torchtext
* pyhanlp
* sanic


## 项目结构
```
kefu_text_similar
├── config.py
├── dict
│   └── stopwords_v2.txt
├── lightnlp_ss.py
├── model.py
├── module.py
├── predict.py
├── README.md
├── requirements.txt
├── ss
├── ss_save
├── start_server.sh
├── text_similar_server.py
├── tool.py
└── utils
    ├── config.py
    ├── cut_words.py
    ├── learning.py
    ├── log.py
    ├── model.py
    ├── module.py
    ├── pad.py
    ├── score_func.py
    ├── tool.py
    └── word_vector.py
```

## 运行
sh start_server.sh
