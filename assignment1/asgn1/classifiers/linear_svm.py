import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg): 
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # calculate dW
        dW[:,j] = dW[:,j] + X[i].T
        dW[:,y[i]] = dW[:, y[i]] - X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # calculate average gradient over training examples, add regularization gradient
  dW = dW / num_train + reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # print 'W: ', W.shape, 'X: ', X.shape, 'y: ', y. shape, 'reg: ', reg
  n_train = X.shape[0]
  n_classes = W.shape[1]

  # get scores for each training example, for each label class
  scores = np.dot(X, W)
    
  # get margins (n_train x n_classes) by 
  # 1. subtracting correct labels (n_train x 1) from scores (n_train x n_classes)
  # 2. seeing if the difference + delta (= 1) is greater or less than 0
  scores_correct = scores[range(n_train), y].reshape(n_train, 1)

  compare_zero = lambda x: max(0, x)
  compare_zero = np.vectorize(compare_zero)

  margins = np.maximum(0, scores - scores_correct + 1)
  margins[range(n_train), y] = 0

  # calculate loss across all training examples, add regularization (penalty l2 norm)
  loss = np.sum(margins) / n_train
  loss_reg = 0.5 * np.sum(W ** 2) * reg 
  loss = loss + loss_reg

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # vectorize this:
  #   for i
  #     for j
  #       if margin > 0:
  #         dW[:,j] = dW[:,j] + X[i].T
  #         dW[:,y[i]] = dW[:, y[i]] - X[i].T
  
  # binarize margins (only get gradient where margin > 0)
  coefs = margins
  coefs[margins > 0] = 1

  # sum of column sums
  coefs[range(n_train), list(y)] = -np.sum(coefs, axis=1)

  # apply coefficients (n_train x n_classes) to all training samples (n_train x number of features)
  dW = np.dot(X.T, coefs)

  # calculate average gradient over training examples, add regularization gradient
  dW = dW / n_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
