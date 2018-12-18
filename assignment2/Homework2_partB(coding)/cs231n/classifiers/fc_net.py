from builtins import range
from builtins import object
import numpy as np
import re

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros(num_classes)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        a1, cache_a1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache_scores = affine_forward(a1, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2)
                                + np.sum(self.params['W2']**2))
        loss = data_loss + reg_loss

        # BP to hidden layer
        dx2, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * self.params['W2']

        # BP to the first layer
        dx, dW1, db1 = affine_relu_backward(dx2, cache_a1)
        dW1 += self.reg * self.params['W1']

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.N = input_dim
        self.C = num_classes
        dims = [self.N] + hidden_dims + [self.C]
        Ws = {'W' + str(i + 1):
              np.random.normal(loc=0.0, scale=weight_scale,size=(dims[i], dims[i+1]))
              for i in range(len(dims)-1)}
        bs = {'b' + str(i + 1): np.zeros(dims[i+1])
             for i in range(len(dims) - 1)}

        self.params.update(Ws)
        self.params.update(bs)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = {}
        if self.normalization is not None:
            if self.normalization == 'batchnorm':
                self.bn_params = {'bn_param' + str(i + 1): \
                                    {'mode': 'train', \
                                    'running_mean': np.zeros(dims[i + 1]), \
                                    'running_var': np.zeros(dims[i + 1])} \
                                    for i in range(self.num_layers - 1)}

                gammas = {'gamma' + str(i + 1): np.ones(dims[i + 1]) \
                        for i in range(self.num_layers - 1)}
                betas = {'beta' + str(i + 1): np.zeros(dims[i + 1]) \
                        for i in range(self.num_layers - 1)}
                self.params.update(gammas)
                self.params.update(betas)
            elif self.normalization=='layernorm':
                self.ln_params = {}
                gammas = {'gamma' + str(i + 1): np.ones(dims[i + 1]) \
                        for i in range(self.num_layers - 1)}
                betas = {'beta' + str(i + 1): np.zeros(dims[i + 1]) \
                        for i in range(self.num_layers - 1)}
                self.params.update(gammas)
                self.params.update(betas)
            else:
                raise ValueError('Unrecognize self.normalization \
                                 mode: %s' %self.normalization)
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for key, bn_param in self.bn_params.items():
                bn_param['mode'] = mode
        if self.normalization=='layernorm':
            ln_param = self.ln_params
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], -1)
        for i in range(self.num_layers):
            # Initialize
            idx = i + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(i)]

            if self.normalization is not None and idx != self.num_layers:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                if self.normalization == 'batchnorm':
                    bn_param = self.bn_params['bn_param' + str(idx)]
                # elif self.normalization == 'layernorm':
                #    ln_param = self.ln_params['ln_param' + str(idx)]

            # Forward
            # last layer
            if idx == self.num_layers:
                h, cache_h = affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            # hidden layers except the last layer
            else:
                if self.normalization == 'batchnorm':
                    h, cache_h = affine_batch_relu_forward(
                        h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                elif self.normalization == 'layernorm':
                    h, cache_h = affine_layer_relu_forward(
                        h, w, b, gamma, beta, ln_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                else:
                    h, cache_h = affine_relu_forward(h, w, b)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h

                if self.use_dropout:
                    h, cache_drop = dropout_forward(h, self.dropout_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                    hidden['cache_drop' + str(idx)] = cache_drop

        scores = hidden['h' + str(self.num_layers)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(self.num_layers):
            idx = i + 1
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(idx)]**2)
        loss = data_loss + reg_loss

        """
        # BP to hidden layer
        dx2, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * self.params['W2']

        # BP to the first layer
        dx, dW1, db1 = affine_relu_backward(dx2, cache_a1)
        dW1 += self.reg * self.params['W1']

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        """

        hidden['dh' + str(self.num_layers)] = dscores
        for i in range(self.num_layers)[::-1]:
            idx = i + 1
            dh = hidden['dh' + str(idx)]
            cache_h = hidden['cache_h' + str(idx)]

            # BP to hidden layers from the last affine layer
            if idx == self.num_layers:
                dh, dw, db = affine_backward(dh, cache_h)
                dw += self.reg * self.params['W' + str(idx)]
                hidden['dh' + str(i)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

            # BP to hidden layers
            else:
                if self.use_dropout:
                    cache_drop = hidden['cache_drop' + str(idx)]
                    dh = dropout_backward(dh, cache_drop)
                if self.normalization is not None:
                    if self.normalization == 'batchnorm':
                        dh, dw, db, dgamma, dbeta = affine_batch_relu_backward(
                            dh, cache_h)
                    elif self.normalization == 'layernorm':
                        dh, dw, db, dgamma, dbeta = affine_layer_relu_backward(
                            dh, cache_h)
                    dw += self.reg * self.params['W' + str(idx)]
                    hidden['dh' + str(i)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = affine_relu_backward(dh, cache_h)
                    dw += self.reg * self.params['W' + str(idx)]
                    hidden['dh' + str(i)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        list_dw, list_db, list_dgamma, list_dbeta = {}, {}, {}, {}
        for key, value in hidden.items():
            if key[:-1] == 'dW':
                list_dw[key[1:]] = value
            if key[:-1] == 'db':
                list_db[key[1:]] = value
            if key[:-1] == 'dgamma':
                list_dgamma[key[1:]] = value
            if key[:-1] == 'dbeta':
                list_dbeta[key[1:]] = value

        # print(hidden.keys(),'\n',list_dw.keys(),'\n',list_db.keys())

        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
