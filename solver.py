import numpy as np
import utils as util
import constants as c
import copy
from sklearn.utils import shuffle as skshuffle

def get_minibatch(X, y, minibatch_size, shuffle=True):
    minibatches = []

    if shuffle:
        X, y = skshuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))
    return minibatches

def sgd(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=16, n_iter=100, print_after=1):
    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            nn.model[layer] -= alpha * grad[layer]

    return nn