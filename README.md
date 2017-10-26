# A simple neural network module for relational reasoning

paper link: https://arxiv.org/abs/1706.01427

## Tensorflow implementation of Relation Network on bAbI dataset

<img src = "./image/relation_network_babi.png" width="650">

## Result

|         |               Accuracy              | Success / Fail |  
|:-------:|:------------------------------------|----------------|
|  Task 1 |                0.994                |        S       |               
|  Task 2 |                0.995               |        S       |              
|  Task 3 |                0.999                |        S       |              
|  Task 4 |                0.956                  |        S       |               
|  Task 5 |                0.999                |        S       |              
|  Task 6 |                0.987                |        S       |              
|  Task 7 |                0.997               |        S       |                 
|  Task 8 |                0.998               |        S       |                
|  Task 9 |                0.868                |        F       |                
| Task 10 |                1               |        S       |               
| Task 11 |                0.998                |        S       |                
| Task 12 |                1                |        S       |                
| Task 13 |                  0.996              |        S       |                
| Task 14 |                1                |        S       |                
| Task 15 |                  0.966              |        S       |               
| Task 16 |                0.483               |        F       |                
| Task 17 |                1                |        S       |               
| Task 18 |                0.999               |        S       |                 
| Task 19 |                1               |        S       |                
| Task 20 |                  1                  |        S       |       

total: **18**/20

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

