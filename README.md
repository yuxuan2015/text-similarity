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
├── getConfig.py
├── lightNLP_server.py
├── lightnlp_ss.py
├── model.py
├── module.py
├── README.md
├── requirements.txt
├── similar_config.ini
├── ss_saves
│   └── cut_clean
│       ├── config.pkl
│       └── model.pkl
├── start_server.sh
├── tool.py
└── utils
    ├── config.py
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
