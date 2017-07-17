# A simple neural network module for relational reasoning

paper link: https://arxiv.org/abs/1706.01427

## Tensorflow implementation of Relation Network on bAbI dataset

<img src = "./image/relation_network_babi.png" width="650">

### Prerequisites

* Python 3.5+ 
* Tensorflow 1.0.1 
* Numpy 
* argparse 
* itertools 
* os 
* pickle 
* re 
* sys 
* datetime 
* time

### Usage

1. **Load data**

```
$ python preprocessing.py --path 'path-where-tasks_1-20_v1-2-located'
```

2. **Run model**

```
$ python train.py
``
