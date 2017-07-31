# A simple neural network module for relational reasoning

paper link: https://arxiv.org/abs/1706.01427

## Tensorflow implementation of Relation Network on bAbI dataset

<img src = "./image/relation_network_babi.png" width="650">

## Result

|         | **with** sentence position |                | **without** sentence position |                |
|:-------:|:-----------------------------------:|:--------------:|:--------------------------------------:|:--------------:|
|         |               Accuracy              | Success / Fail |                Accuracy                | Success / Fail |
|  Task 1 |                0.999                |        S       |                  0.999                 |        S       |
|  Task 2 |                0.8422               |        F       |                 0.7277                 |        F       |
|  Task 3 |                0.802                |        F       |                 0.7447                 |        F       |
|  Task 4 |                  1                  |        S       |                    1                   |        S       |
|  Task 5 |                0.992                |        S       |                 0.9839                 |        S       |
|  Task 6 |                0.996                |        S       |                  0.997                 |        S       |
|  Task 7 |                0.9869               |        S       |                  0.994                 |        S       |
|  Task 8 |                0.9719               |        S       |                 0.9829                 |        S       |
|  Task 9 |                0.998                |        S       |                  0.997                 |        S       |
| Task 10 |                0.9899               |        S       |                 0.9709                 |        S       |
| Task 11 |                0.994                |        S       |                 0.9869                 |        S       |
| Task 12 |                0.999                |        S       |                 0.9749                 |        S       |
| Task 13 |                  1                  |        S       |                 0.9608                 |        S       |
| Task 14 |                0.997                |        S       |                  0.997                 |        S       |
| Task 15 |                  1                  |        S       |                 0.6576                 |        F       |
| Task 16 |                0.4324               |        F       |                 0.4494                 |        F       |
| Task 17 |                0.876                |        F       |                 0.7984                 |        F       |
| Task 18 |                0.9669               |        S       |                 0.9719                 |        S       |
| Task 19 |                0.6026               |        F       |                 0.4845                 |        F       |
| Task 20 |                  1                  |        S       |                    1                   |        S       |

total: **15**/20 (with sentence position), **14**/20 (without sentence position)

### Prerequisites

* Python 3.5+ 
* Tensorflow-gpu 1.1+
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

