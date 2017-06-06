import numpy as np
from random import shuffle

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
  n_classes = W.shape[1]
  n_train = X.shape[0]
  loss = 0.0

  for i in range(n_train):
    # get the scores for each class by applying weights to training sample
    scores = np.dot(X[i], W)
    # help stabilize numbers so that the highest number is 0
    scores_shift = scores - np.max(scores)

    # compute stuff for loss
    # sum over all j: e ^ f_j
    sum_scores = 0.0
    for j in range(n_classes):
      sum_scores += np.exp(scores_shift[j])

    # loss for training sample at i = - f_y_i + log ( sum over all j: e ^ f_j )
    loss_i = -scores_shift[y[i]] + np.log(sum_scores)
    loss += loss_i

    # compute stuff for gradient
    # p of y_i given x_i and weights W = e ^ f_y_i / sum over all j: e ^ f_j
    for j in range(n_classes):
      p = np.exp(scores_shift[j]) / sum_scores
      # if its the right class
      if j == y[i]:
        dW[:, j] += (p - 1) * X[i]
      else:
        dW[:, j] += p * X[i]

  # get average loss and gradient over all training samples, adding regularization
  loss /= n_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW = dW / n_train + reg * W 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  
  n_classes = W.shape[1]
  n_train = X.shape[0]

  # get the scores for each class by applying weights to training sample
  scores = np.dot(X, W)
  # help stabilize numbers so that the highest number is 0
  scores_shift = scores - np.max(scores)

  # calculate "loss" which is really normalizing scores
  exp = np.exp(scores_shift)
  exp_summed = np.sum(exp, axis = 1).reshape(n_train, 1)
  softmax = exp / exp_summed
  
  # cross entropy loss across all training examples
  loss = -np.sum(np.log(softmax[range(n_train), y])) / n_train

  # compute the gradient w.r.t. scores
  idxs = np.zeros([n_train, n_classes])
  idxs[range(n_train), y] = 1
  dX = softmax - idxs
  # get gradient across all training examples
  dW = np.dot(dX.T, X).T / n_train

  # add regularization to loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W 

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

