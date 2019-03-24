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



## 4 - Neural Network Model (Non-linear)

### 4.1 preparation

* Picture of model:

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/3.jpg)

* Mathematically:

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/4.jpg)


* cost function $J$			

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/5.jpg)

### 4.2 Implement(4 hidden units)

#### 4.2.1 Initialize parameters

* ==Arguments== : 

  function `layer_sizes(X, Y)` return (n_x, n_h, n_y)

  * n_x -- size of the input layer
  * n_h -- size of the hidden layer (initialize to 4)
  * n_y -- size of the output layer

  

* ==instructions== : 

  function `initialize_parameters(n_x, n_h, n_y)` return dictionary parameters

  * W1-- (n_h, n_x) matrix with random values 
  * b1 -- (n_h, 1) vector with zeros
  * W2 -- (n_y, n_h) matrix with random values 
  * b2 -- (n_y, 1) vector with zero

#### 4.2.2 NN_Model in Loop: `nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False)`

* ==forward propagation==  : 

  function `forward_propagation(X, parameters)` return A2, dictionary cache

  * Z1-- np.dot(W1, X) + b1
  * A1 -- np.tanh(Z1)
  * Z2 -- np.dot(W2, A1) + b2
  * A2 -- sigmoid(Z2)

* ==compute loss== : 

  function `compute_cost(A2, Y, parameters)` return cost

  * m = Y.shape[1]
  * logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2), (1 - Y))
  * cost = -np.sum(logprobs) / m

* ==backward propagation== to get the gradients

  using backward propagation to get the gradient to implement gradient descent algorithm

  ![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/6.jpg)

  function `backward_propagation(parameters, cache, X, Y)` return dictionary grads of dW1, db1, dW2, db2

  *  m --  X.shape[1]
  * dZ2  --  A2 - Y
  * dW2 -- np.dot(dZ2, A1.T) / m
  * db2 -- np.sum(dZ2, axis = 1, keepdims = True) / m
  * dZ1 -- np.dot(W2.T, dZ2) *  (1 - np.power(A1, 2))
  * dW1 -- np.dot(dZ1, X.T) / m
  * db1 -- np.sum(dZ1, axis = 1, keepdims = True) / m

* ==update parameters==(gradient descent rule)

  ![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/7.jpg)

  * new weight -- weight - learning_rate * dweight
  * new bias -- bias - learning_rate * dbias

#### 4.2.3 Predictions `predict(parameters, X)`

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/8.jpg)

* Result of cost : 

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/9.jpg)

After 10 iterations, the cost function has reduced to 0.218612 and accurancy of predictions is 90%



#### 4.2.4 Different hidden layer size

with for loop of hidden layer size [1, 2, 3, 4, 5, 20, 50]

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/10.jpg)

As above, predictions has the highest accurancy 91.25% with 5 hidden units

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/11.jpg)

![](https://github.com/santochaoya/Deep-Learning-with-Nerual-Network/blob/master/binary%20classification%20with%20tanh/12.jpg)





---

Reference:

Coursera week 3 -- Planar data classification with one hidden layer v5

