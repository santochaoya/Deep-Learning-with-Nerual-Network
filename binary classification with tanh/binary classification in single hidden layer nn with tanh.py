# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# import dataset, X contains features x1 and x2 each contains 400 training examples, Y contains labels
# label : 0 - red, 1 - blue
X, Y = load_planar_dataset()
T = Y.reshape(400,)

#Visualize the data:
plt.scatter(X[0, :], X[1, :], c=T, s=40, cmap=plt.cm.Spectral)# scatter : plot single point

#check shape of X, Y, and size of training examples
shape_X = X.shape
shape_Y = Y.shape
m = np.size(Y)  # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, T)
plt.title("Logistic Regression")
plt.show()


# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")




