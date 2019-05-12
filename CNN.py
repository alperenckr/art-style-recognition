import numpy as np
import loss as loss_fun
import layers as l
import regularization as reg
import utils as util


class NeuralNet(object):
    """
    Ana nöron ağı sınıfı
    Diğer nöron ağı sınıfları bu sınıftan türetilir.
    Bu kodda bir tane nöron ağı çeşidi bulunmaktadır.
    """

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
    """
    Konvolüsyonel sinir ağı
    NeuralNet sınıfından türetilir

    c: etiket sayısı
    d: input size


    """
    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # Conv-1
        # input: 227 x 227 stride: 4 pad: 0
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'],stride=4,padding=0)
        h1, nl_cache1 = l.relu_forward(h1)
        # output: 



        # Pool-1
        hpool, hpool_cache = l.maxpool_forward(h1,size = 3, stride = 2)
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



class OurConvNet(NeuralNet):
    
    def __init__(self):
        self.model = self._init_model()

    def forward(self, train_data):
        # X (image_count, RGB, width,height)
        # I == image_count
        #conv_forward(input: Ix3x224x224, filter_size: 96 x 1 x 11 x 11 , stride: 4, padding: 2)output: Ix96x55x55
        #maxpool_forward(input: Ix96x55x55, filter_size: 3x3, stride: 2)output: Ix96x27x27
        #conv_forward(input: Ix96x27x27, filter_size: 256x1x5x5, stride: 1, padding: 2)output: Ix256x27x27
        #maxpool_forward(input: Ix256x27x27, filter_size: 3x3, stride: 2)output: Ix256x13x13
        #conv_forward(input: Ix256x13x13, filter_size: 384x1x3x3, stride: 1, padding: 1)output: Ix384x13x13
        #conv_forward(input: Ix384x13x13, filter_size: 384x1x3x3, stride: 1, padding: 1)output: Ix384x13x13
        #conv_forward(input: Ix384x13x13, filter_size: 256x1x3x3, stride: 1, padding: 1)output: Ix256x13x13
        #maxpool_forward(input: Ix256x13x13, filter_size: 3x3, stride: 2)output: Ix256x6x6
        #fc_forward(input: Ix9216)output: Ix4096
        #fc_forward(input: Ix4096)output: Ix1000
        cache = {}
        out1, cache1 = l.conv_forward(train_data, self.model["W1"], self.model["b1"], stride=4,padding=2) #output image_cnt x 96 x 55 x 55
        cache["conv1"] = cache1

        bn_cache1 = (self.bn_caches["bn1_mean"], self.bn_caches["bn1_var"])
        bn1, bn_cache1, run_mean, run_var = l.bn_forward(out1, self.model["gamma1"], self.model["beta1"], bn_cache1)
        self.bn_caches["bn1_mean"], self.bn_caches["bn1_var"] = run_mean, run_var
        cache["bn1"] = bn_cache1
        
        relu1, relu_cache1 = l.relu_forward(bn1)
        cache["relu1"] = relu_cache1

        hpool1, hpool_cache1 = l.maxpool_forward(relu1,size = 3, stride = 2)             #output: Ix96x27x27
        cache["pool1"] = hpool_cache1

        out2, cache2 = l.conv_forward(hpool1, self.model["W2"], self.model["b2"], stride=1,padding=2)     #output: Ix256x27x27
        cache["conv2"] = cache2

        bn_cache2 = (self.bn_caches["bn2_mean"], self.bn_caches["bn2_var"])
        bn2, bn_cache2, run_mean, run_var = l.bn_forward(out2, self.model["gamma2"], self.model["beta2"], bn_cache2)
        self.bn_caches["bn2_mean"], self.bn_caches["bn2_var"] = run_mean, run_var
        cache["bn2"] = bn_cache2

        relu2, relu_cache2 = l.relu_forward(bn2)
        cache["relu2"] = relu_cache2
        
        hpool2, hpool_cache2 = l.maxpool_forward(relu2, size = 3, stride = 2)             #output: Ix256x13x13
        cache["pool2"] = hpool_cache2

        out3, cache3 = l.conv_forward(hpool2, self.model["W3"], self.model["b3"], stride=1,padding=1)     #output: Ix384x13x13
        cache["conv3"] = cache3

        bn_cache3 = (self.bn_caches["bn3_mean"], self.bn_caches["bn3_var"])
        bn3, bn_cache3, run_mean, run_var = l.bn_forward(out3, self.model["gamma3"], self.model["beta3"], bn_cache3)
        self.bn_caches["bn3_mean"], self.bn_caches["bn3_var"] = run_mean, run_var
        cache["bn3"] = bn_cache3

        relu3, relu_cache3 = l.relu_forward(bn3)
        cache["relu3"] = relu_cache3

        out4, cache4 = l.conv_forward(relu3, self.model["W4"], self.model["b4"], stride=1,padding=1)     #output: Ix384x13x13
        cache["conv4"] = cache4

        bn_cache4 = (self.bn_caches["bn4_mean"], self.bn_caches["bn4_var"])
        bn4, bn_cache4, run_mean, run_var = l.bn_forward(out4, self.model["gamma4"], self.model["beta4"], bn_cache4)
        self.bn_caches["bn4_mean"], self.bn_caches["bn4_var"] = run_mean, run_var
        cache["bn4"] = bn_cache4

        relu4, relu_cache4 = l.relu_forward(bn4)
        cache["relu4"] = relu_cache4

        out5, cache5 = l.conv_forward(relu4, self.model["W5"], self.model["b5"], stride=1,padding=1)     #output: Ix256x13x13
        cache["conv5"] = cache5
        
        bn_cache5 = (self.bn_caches["bn5_mean"], self.bn_caches["bn5_var"])
        bn5, bn_cache5, run_mean, run_var = l.bn_forward(out5, self.model["gamma5"], self.model["beta5"], bn_cache5)
        self.bn_caches["bn5_mean"], self.bn_caches["bn5_var"] = run_mean, run_var
        cache["bn5"] = bn_cache5
        
        relu5, relu_cache5 = l.relu_forward(bn5)
        cache["relu5"] = relu_cache5

        hpool3, hpool_cache3 = l.maxpool_forward(relu5,size = 3, stride = 2)             #output: Ix256x6x6
        cache["pool3"] = hpool_cache3
        
        cache["reshape_pool"] = hpool3
        h = hpool3.ravel().reshape(train_data.shape[0], -1)
        

        out6, cache6 = l.fc_forward(h, self.model['W6'], self.model['b6'])         #output: Ix4096
        cache["fc1"] = cache6

        sig1, cache_sig1 = l.sigmoid_forward(out6)
        cache["sig1"] = cache_sig1

        out7, cache7 = l.fc_forward(sig1, self.model['W7'], self.model['b7'])           #output: Ix1000
        cache["fc2"] = cache7
        
        sig2, cache_sig2 = l.sigmoid_forward(out7)
        cache["sig2"] = cache_sig2
        
        out8, cache8 = l.fc_forward(sig2, self.model['W8'], self.model['b8'])           #output: # of classes
        cache["fc3"] = cache8

        cache["output"] = out8
        
        return cache 

    def backward(self, cache, y_train):
        
        
        grad_out = loss_fun.dcross_entropy(cache["output"],y_train)

        grad = {}

        dp8, dW8, db8 = l.fc_backward(grad_out, cache["fc3"])
        grad["W8"] = dW8
        grad["b8"] = db8

        dsig2 = l.sigmoid_backward(dp8, cache["sig2"])

        dp7, dW7, db7 = l.fc_backward(dsig2, cache["fc2"])
        grad["W7"] = dW7
        grad["b7"] = db7

        dsig1 = l.sigmoid_backward(dp7, cache["sig1"])

        dp6, dW6, db6 = l.fc_backward(dsig1, cache["fc1"])
        grad["W6"] = dW6
        grad["b6"] = db6        
        dp6 = dp6.ravel().reshape(cache["reshape_pool"].shape)

        dpool3 = l.maxpool_backward(dp6, cache["pool3"])

        drelu5 = l.relu_backward(dpool3, cache["relu5"])

        dbn5, dgamma5, dbeta5 = l.bn_backward(drelu5, cache["bn5"])
        grad["gamma5"] = dgamma5
        grad["beta5"] = dbeta5

        dp5, dW5, db5 = l.conv_backward(dbn5, cache["conv5"])
        grad["W5"] = dW5
        grad["b5"] = db5
        drelu4 = l.relu_backward(dp5, cache["relu4"])

        dbn4, dgamma4, dbeta4 = l.bn_backward(drelu4, cache["bn4"])
        grad["gamma4"] = dgamma4
        grad["beta4"] = dbeta4

        dp4, dW4, db4 = l.conv_backward(drelu4, cache["conv4"])
        grad["W4"] = dW4
        grad["b4"] = db4
        drelu3 = l.relu_backward(dp4, cache["relu3"])

        dbn3, dgamma3, dbeta3 = l.bn_backward(drelu3, cache["bn3"])
        grad["gamma3"] = dgamma3
        grad["beta3"] = dbeta3

        dp3, dW3, db3 = l.conv_backward(drelu3, cache["conv3"])
        grad["W3"] = dW3
        grad["b3"] = db3
        dpool2 = l.maxpool_backward(dp3, cache["pool2"])

        drelu2 = l.relu_backward(dpool2, cache["relu2"])

        dbn2, dgamma2, dbeta2 = l.bn_backward(drelu2, cache["bn2"])
        grad["gamma2"] = dgamma2
        grad["beta2"] = dbeta2

        dp2, dW2, db2 = l.conv_backward(drelu2, cache["conv2"])
        grad["W2"] = dW2
        grad["b2"] = db2
        dpool1 = l.maxpool_backward(dp2, cache["pool1"])

        drelu1 = l.relu_backward(dpool1, cache["relu1"])

        dbn1, dgamma1, dbeta1 = l.bn_backward(drelu1, cache["bn1"])
        grad["gamma1"] = dgamma1
        grad["beta1"] = dbeta1

        dp1, dW1, db1 = l.conv_backward(drelu1, cache["conv1"])
        grad["W1"] = dW1
        grad["b1"] = db1

        return grad

    def _init_model(self):
        #ideal way of initialize for relu activation function
        mu, sigma = 0, 1 #mean and standart variation
        model = dict(
            W1=np.random.normal(mu, sigma, (96,3, 11, 11)),
            W2=np.random.normal(mu, sigma, (256,96, 5, 5)),
            W3=np.random.normal(mu, sigma, (384,256, 3, 3)),
            W4=np.random.normal(mu, sigma, (384,384, 3, 3)),
            W5=np.random.normal(mu, sigma, (256,384, 3, 3)),
            W6=np.random.normal(mu, sigma, (9216, 4096)),
            W7=np.random.normal(mu, sigma, (4096, 4096)),
            W8=np.random.normal(mu, sigma, (4096, 6)),
            b1=np.zeros((96, 1)),
            b2=np.zeros((256, 1)),
            b3=np.zeros((384,1)),
            b4=np.zeros((384,1)),
            b5=np.zeros((256,1)),
            b6=np.zeros((1,4096)),
            b7=np.zeros((1,4096)),
            b8=np.zeros((1,6)),
            gamma1=np.ones((16, 96, 55, 55)),
            gamma2=np.ones((16, 256, 27, 27)),
            gamma3=np.ones((16, 384, 13, 13)),
            gamma4=np.ones((16, 384, 13, 13)),
            gamma5=np.ones((16, 256,13,13)),
            beta1 = np.zeros((16, 96, 55, 55)),
            beta2 = np.zeros((16, 256, 27, 27)),
            beta3 = np.zeros((16, 384, 13, 13)),
            beta4 = np.zeros((16, 384, 13, 13)),
            beta5 = np.zeros((16, 256,13,13))
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((96, 55, 55)),
            bn2_mean=np.zeros((256, 27, 27)),
            bn3_mean=np.zeros((384, 13, 13)),
            bn4_mean=np.zeros((384, 13, 13)),
            bn5_mean=np.zeros((256,13,13)),
            bn1_var=np.zeros((96, 55, 55)),
            bn2_var=np.zeros((256, 27, 27)),
            bn3_var=np.zeros((384, 13, 13)),
            bn4_var=np.zeros((384, 13, 13)),
            bn5_var=np.zeros((256,13,13))
        )
        return model

    def train_step(self, X_train, y_train):
        cache = self.forward(X_train)
        loss = loss_fun.cross_entropy(self.model, cache["output"], y_train)
        grad = self.backward(cache, y_train)
        return grad, loss
