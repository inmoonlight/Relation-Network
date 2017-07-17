# A simple neural network module for relational reasoning

paper link: https://arxiv.org/abs/1706.01427

## Tensorflow implementation of Relation Network on bAbI dataset

<img src = "./image/relation_network_babi.png" width="650">

## Result

|   Task  | Accuracy | Success/Fail |
|:-------:|:--------:|:------------:|
| Task 1  |   0.9990 |       S      |
| Task 2  |   0.8422 |       F      |
| Task 3  |   0.8020 |       F      |
| Task 4  |   1.0000 |       S      |
| Task 5  |   0.9920 |       S      |
| Task 6  |   0.9960 |       S      |
| Task 7  |   0.9869 |       S      |
| Task 8  |   0.9719 |       S      |
| Task 9  |   0.9980 |       S      |
| Task 10 |   0.9899 |       S      |
| Task 11 |   0.9940 |       S      |
| Task 12 |   0.9990 |       S      |
| Task 13 |   1.0000 |       S      |
| Task 14 |   0.9970 |       S      |
| Task 15 |   1.0000 |       S      |
| Task 16 |   0.4324 |       F      |
| Task 17 |   0.8760 |       F      |
| Task 18 |   0.9669 |       S      |
| Task 19 |   0.6026 |       F      |
| Task 20 |   1.0000 |       S      |

total: **15**/20

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
```

