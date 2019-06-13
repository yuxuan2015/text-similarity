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
├── data
│   ├── atec_nlp_sim_train_add.csv
│   └── atec_nlp_sim_train.csv
├── data_utils.py
├── dict
│   ├── atec_words.txt
│   └── stop_words.txt
├── lightnlp_ss.py
├── model.py
├── module.py
├── predict.py
├── README.md
├── requirements.txt
├── ss
├── ss_saves
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


## Reference

1 [lightnlp](https://github.com/smilelight/lightNLP)

2 [ATEC-NLP](https://github.com/yuxuan2015/ATEC-NLP)