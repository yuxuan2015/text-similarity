## 工程介绍
客服问题相似度计算，是贯穿智能客服离线、在线和运营等几乎所有环节最核心的技术，同时也是自然语言理解中最核心的问题之一，广泛应用于搜索、推荐、对话等领域。


## 模块介绍
|项目名称 |  负责人 | 应用说明 |
|--|--|---|
|客服文本相似性计算|娄源源|nlp服务      |    



## 环境依赖
* python3.6
* torch


## 快速访问
|   应用名称 | 环境|  访问地址 |账号|
|--------|-------|---------------|-
|kefu_text_similar|测试|无 |无



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
