import numpy as np
import hipsternet.loss as loss_fun
import layers as l
import regularization as reg
import utils as util


class NeuralNet(object):

    loss_funs = dict(
        cross_ent=loss_fun.cross_entropy,
        hinge=loss_fun.hinge_loss,
        squared=loss_fun.squared_loss,
        l2_regression=loss_fun.l2_regression,
        l1_regression=loss_fun.l1_regression
    )

    dloss_funs = dict(
        cross_ent=loss_fun.dcross_entropy,
        hinge=loss_fun.dhinge_loss,
        squared=loss_fun.dsquared_loss,
        l2_regression=loss_fun.dl2_regression,
        l1_regression=loss_fun.dl1_regression
    )

    forward_nonlins = dict(
        relu=l.relu_forward,
        lrelu=l.lrelu_forward,
        sigmoid=l.sigmoid_forward,
        tanh=l.tanh_forward
    )

    backward_nonlins = dict(
        relu=l.relu_backward,
        lrelu=l.lrelu_backward,
        sigmoid=l.sigmoid_backward,
        tanh=l.tanh_backward
    )

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        if loss not in NeuralNet.loss_funs.keys():
            raise Exception('Loss function must be in {}!'.format(NeuralNet.loss_funs.keys()))

        if nonlin not in NeuralNet.forward_nonlins.keys():
            raise Exception('Nonlinearity must be in {}!'.format(NeuralNet.forward_nonlins.keys()))

        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss
        self.forward_nonlin = NeuralNet.forward_nonlins[nonlin]
        self.backward_nonlin = NeuralNet.backward_nonlins[nonlin]
        self.mode = 'classification'

        if 'regression' in loss:
            self.mode = 'regression'

    def train_step(self, X_train, y_train):
        """
        Single training step over minibatch: forward, loss, backprop
        """
        y_pred, cache = self.forward(X_train, train=True)
        loss = self.loss_funs[self.loss](self.model, y_pred, y_train, self.lam)
        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def predict_proba(self, X):
        score, _ = self.forward(X, False)
        return util.softmax(score)

    def predict(self, X):
        if self.mode == 'classification':
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            score, _ = self.forward(X, False)
            y_pred = np.round(score)
            return y_pred

    def forward(self, X, train=False):
        raise NotImplementedError()

    def backward(self, y_pred, y_train, cache):
        raise NotImplementedError()

    def _init_model(self, D, C, H):
        raise NotImplementedError()

class ConvNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # Conv-1
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = l.relu_forward(h1)

        # Pool-1
        hpool, hpool_cache = l.maxpool_forward(h1)
        h2 = hpool.ravel().reshape(X.shape[0], -1)

        # FC-7
        h3, h3_cache = l.fc_forward(h2, self.model['W2'], self.model['b2'])
        h3, nl_cache3 = l.relu_forward(h3)

        # Softmax
        score, score_cache = l.fc_forward(h3, self.model['W3'], self.model['b3'])

        return score, (X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3)

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3 = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # FC-7
        dh3, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dh3 = self.backward_nonlin(dh3, nl_cache3)

        dh2, dW2, db2 = l.fc_backward(dh3, h3_cache)
        dh2 = dh2.ravel().reshape(hpool.shape)

        # Pool-1
        dpool = l.maxpool_backward(dh2, hpool_cache)

        # Conv-1
        dh1 = self.backward_nonlin(dpool, nl_cache1)
        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(D * 14 * 14, H) / np.sqrt(D * 14 * 14 / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((D, 1)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C))
        )


