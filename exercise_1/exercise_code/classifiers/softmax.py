"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    N = # images;
    D = 32 * 32 * 3 + 1
    C = 10

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (32 * 32 * 3 + 1 , 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (# images , 32 * 32 * 3 + 1)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means (# images)
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # iterate over images
    classes = W.shape[1]
    images = X.shape[0]
    
    for imageNumber in range(images):
        scores = X[imageNumber].dot(W)
        
        exps = np.exp(scores - np.max(scores))
        exps /= np.sum(exps)
                
        loss -= np.log(exps[y[imageNumber]])

        exps[y[imageNumber]] -= 1
        
        for clazz in range(classes):
            dW[:,clazz] += X[imageNumber] * exps[clazz]    
        
    loss /= images
    dW /= images
    
    loss += reg * np.sum(W * W)
    dW += reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    images = X.shape[0]
    scores = X.dot(W)
    exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    exps /= np.sum(exps, axis=1, keepdims=True)
                
    loss = np.sum(-np.log(exps[range(images), y])) / images
    loss += reg * np.sum(W * W)
    
    exps[range(images), y] -= 1
    dW = X.T.dot(exps) / images
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)
        
def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    import time

    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-1]
    regularization_strengths = [1e-2]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    iteration = 0;
    learning_rate_range = learning_rates #np.arange(learning_rates[0], learning_rates[1], learning_rates[0])
    regularization_strength_range = regularization_strengths #np.arange(regularization_strengths[0], regularization_strengths[1], regularization_strengths[0])
    all_iterations = len(learning_rate_range) * len(regularization_strength_range)
    
    all_time = 0.0
        
    for learning_rate in learning_rate_range:
        for regularization_strength in regularization_strength_range:
            print('Starting new iteration')
            tic = time.time()
            softmax = SoftmaxClassifier()
            print('Training')
            softmax.train(X_train, y_train, learning_rate=learning_rate, reg=regularization_strength, num_iters=15000);
            
            print('Predicting')
            y_pred_train = softmax.predict(X_train)
            y_pred_val = softmax.predict(X_val)
            
            training_accuracy = np.mean(y_train == y_pred_train)
            validation_accuracy = np.mean(y_val == y_pred_val)

            results[(learning_rate, regularization_strength)] = (training_accuracy, validation_accuracy)
            
            if validation_accuracy > best_val:
                print(f'New best - Training acc: {training_accuracy} - Validation acc: {validation_accuracy} ')
                best_val = validation_accuracy
                best_softmax = softmax
            all_classifiers.append((softmax, validation_accuracy)) 
            toc = time.time()
            duration = toc - tic
            iteration += 1
            all_time += duration
            print(f'Iteration {iteration} of {all_iterations} took {duration}')
            print(f'Average time: {all_time / iteration}')
            print(f'Total time: {all_time}')
            print(f'Remaining time: {(all_time / iteration) * (all_iterations - iteration)}')
            
    
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
