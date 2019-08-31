# 基于BERT的电商评论观点挖掘
## 实现说明
基于pytorch和bert，参考NER的网络结构，第一个全连接的输出层输出属性/观点的起始位置（按照BEIS，共有8个类），第二个输出观点+属性的分类（合计有28个类）

## 运行环境
python3.6 + pytorch==1.0.1 + pytorch-pretrained-bert==0.6.2

## 运行命令
cd code; CUDA_VISIBLE_DEVICES=0 python  train_v2.py

## 比赛链接
[电商评论观点挖掘-天池大赛-阿里云天池](https://tianchi.aliyun.com/competition/entrance/231731/ )
