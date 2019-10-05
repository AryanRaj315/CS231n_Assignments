from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Hint as given in cs231n documentation 
    # f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
    # p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup
    # instead: first shift the values of f so that the highest number is 0:
    # f -= np.max(f) # f becomes [-666, -333, 0]
    # p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer


    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(X.shape[0]):
        f = X[i].dot(W)
        f -= np.max(f)
        p = np.exp(f)/(np.sum(np.exp(f)))
        loss += -np.log(p[y[i]])

    num_classes = W.shape[0]
    num_train = X.shape[1]
    for i in range(num_train):
        for j in range(num_classes):
            f = X[j].dot(W)
            f -= np.max(f)
            p = np.exp(f)/(np.sum(np.exp(f)))
            dW[j, :] += (p-(j == y[i])) * X[:, i]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
