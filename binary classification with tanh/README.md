# Binary classification with one hidden layer

## 1 - Packages

* numpy
* sklearn
* matplotlib
* self-import : testCases, planar_utils

```python
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent
```

## 2 - Dataset

Get the dataset from testCase_v2 into X and Y.

* X contains features X1, X2 each feature has 400 training examples
* Y containts labels of 0 - red, 1 - blue

Plot dataset in axis of X1 and X2 with color label of Y

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/1.jpg)

## 3 - Logistic Regression

First, fit data with logistic regression and plot the decision boundary.

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/2.jpg)

As the accuracy of logistic regression is only 47%, logistic regression performed not good on this dataset of a binary classification.



## 4 - Neural Network Model

### 4.1 preparation

* Picture of model:

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/3.jpg)

* Mathematically:

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/4.jpg)



* cost function $J$			
  $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
  

### 4.2 Implement

* Initialize the model's parameters
* Loop:
  * forward propagation
  * compute loss
  * backward porpagation to get the gradients
  * update parameters(gradient descent)

