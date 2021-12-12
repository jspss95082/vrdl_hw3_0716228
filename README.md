---
title: 'VRDL HW3 0716228'
disqus: hackmd
---

VRDL HW3 0716228
===


[![hackmd-github-sync-badge](https://hackmd.io/ldmMEtXKQkWkRNVQKxOnMA/badge)](https://hackmd.io/ldmMEtXKQkWkRNVQKxOnMA)



## Table of Contents

[TOC]

## How to generate answer.json?

1.  git clone this project
2.  Download train dataset  from [here](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view) 
and unzip it
3.  Download pretrained model from [here](https://drive.google.com/file/d/1RV2JBMsNR7F4B0bO6EHnBnFxOehVgzgd/view?usp=sharing) or train your own model and name it as `model.pt`
4.  `pip install -r requirements.txt`
5.  run all cell and you can get answer.json

## How to train the model
1.  git clone this project
2.  Download train dataset  from [here](https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view) 
and unzip it 
3.  `pip install -r requirements.txt`
4.  `python train.py --epoch 1900 --batch-size 4`
5.  `model.pt` is your model

## Pretrained model link
[link](https://drive.google.com/file/d/1RV2JBMsNR7F4B0bO6EHnBnFxOehVgzgd/view?usp=sharing)



#### only `inference.ipynb` and `train.py` were wirtten by me, so others don't follow PEP8

###### tags: `VRDL` `Mask Rcnn`