import numpy as np

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])




def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s


def propagate(w, b, X, Y):


    m = X.shape[1]


      # compute activation


    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m)  # compute cost

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))