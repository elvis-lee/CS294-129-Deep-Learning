import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *
from deeplearning.classifiers.fc_net import *

class HaoConvNet4(object):
  """
  [conv-spatialbathnorm-relu]x2 - [conv-spatialbathnorm-relu-pool]x3 - [affine-bathnorm-relu] - dropout - affine - softmax
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=16, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dropout=0.5, dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    input_c, input_h, input_w = input_dim

    #conv layer
    #input size (N, C, H, W)
    #output size (N,number_filters, H, W)
    self.params['W0'] = np.random.normal(scale = weight_scale, size = num_filters*input_c*filter_size**2).reshape((num_filters, input_c, filter_size, filter_size))
    self.params['b0'] = np.zeros(num_filters)
    self.params['gamma0'] = np.ones(num_filters)
    self.params['beta0'] = np.zeros(num_filters)

    #conv layer
    #input size (N, C, H, W)
    #output size (N,number_filters, H, W)
    self.params['W1'] = np.random.normal(scale = weight_scale, size = num_filters**2*filter_size**2).reshape((num_filters, num_filters, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)

    #conv layer
    #input size (N, C, H, W)
    #output size (N,number_filters, H, W)
    self.params['W2'] = np.random.normal(scale = weight_scale, size = num_filters**2*filter_size**2).reshape((num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    
    #pooling layer
    #pool_param is defined in the loss function
    #input size (N, number_filters, H, W) 
    #output size (N, number_filters, H/2, W/2)


    #conv layer
    #input size (N, number_filters, H/2, W/2)
    #output size (N,number_filters, H/2, W/2)
    #(H_out, W_out) = (H, W) because of the padding and stride config in the loss function, 
    self.params['W3'] = np.random.normal(scale = weight_scale, size = num_filters**2*filter_size**2).reshape((num_filters, num_filters, filter_size, filter_size))
    self.params['b3'] = np.zeros(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta3'] = np.zeros(num_filters)

    #pooling layer
    #pool_param is defined in the loss function
    #input size (N, number_filters, H/2, W/2)
    #output size (N,number_filters, H/4, W/4)


    #conv layer
    #input size (N, number_filters, H/4, W/4)
    #output size (N,number_filters, H/4, W/4)
    #(H_out, W_out) = (H, W) because of the padding and stride config in the loss function, 
    self.params['W4'] = np.random.normal(scale = weight_scale, size = (num_filters**2)*(filter_size**2)).reshape((num_filters, num_filters, filter_size, filter_size))
    self.params['b4'] = np.zeros(num_filters)
    self.params['gamma4'] = np.ones(num_filters)
    self.params['beta4'] = np.zeros(num_filters)
    #pooling layer
    #pool_param is defined in the loss function
    #input size (N, number_filters, H/4, W/4) 
    #output size (N, number_filters, H/8, W/8)


    #hidden affine layer
    #input size (N, number_filters, H/8, W/8)
    #output size (N, hidden_dim)
    self.params['W5'] = np.random.normal(scale = weight_scale, size = num_filters*(input_h/8)*(input_w/8)*hidden_dim).reshape((num_filters*(input_h/8)*(input_w/8),hidden_dim))
    self.params['b5'] = np.zeros(hidden_dim)
    self.params['gamma5'] = np.ones(hidden_dim)
    self.params['beta5'] = np.zeros(hidden_dim)

    #dropout layer


    #output affine layer
    #input size (N, hidden_dim)
    #output size (N, num_classes)
    self.params['W6'] = np.random.normal(scale = weight_scale, size = hidden_dim*num_classes).reshape((hidden_dim, num_classes))
    self.params['b6'] = np.zeros(num_classes)

    

    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in np.arange(6)]

    self.dropout_param = {}
    self.dropout_param = {'mode': 'train', 'p': dropout}
      

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)




  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    mode = 'test' if y is None else 'train'

    for bn_param in self.bn_params:
        bn_param[mode] = mode
    self.dropout_param['mode'] = mode   

    W0, b0 = self.params['W0'], self.params['b0']
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    gamma0 = self.params['gamma0']
    gamma1 = self.params['gamma1']
    gamma2 = self.params['gamma2']
    gamma3 = self.params['gamma3']
    gamma4 = self.params['gamma4']
    gamma5 = self.params['gamma5']
    beta0 = self.params['beta0']
    beta1 = self.params['beta1']
    beta2 = self.params['beta2']
    beta3 = self.params['beta3']
    beta4 = self.params['beta4']
    beta5 = self.params['beta5']
    bn_param0 = self.bn_params[0]
    bn_param1 = self.bn_params[1]
    bn_param2 = self.bn_params[2]
    bn_param3 = self.bn_params[3]
    bn_param4 = self.bn_params[4]
    bn_param5 = self.bn_params[5]

    out, cache0 = conv_spabachnorm_relu_forward(X, W0, b0, conv_param, bn_param0, gamma0, beta0)
    out, cache1 =  conv_spabachnorm_relu_forward(out, W1, b1, conv_param, bn_param1, gamma1, beta1)

    out, cache2 = conv_spabachnorm_relu_pool_forward(out, W2, b2, conv_param, bn_param2, gamma2, beta2, pool_param)
    out, cache3 = conv_spabachnorm_relu_pool_forward(out, W3, b3, conv_param, bn_param3, gamma3, beta3, pool_param)
    out, cache4 = conv_spabachnorm_relu_pool_forward(out, W4, b4, conv_param, bn_param4, gamma4, beta4, pool_param)

    out, cache5 = affine_batnorm_relu_forward(out, W5, b5, gamma5, beta5, bn_param5)
    out, do_cache = dropout_forward(out, self.dropout_param)
    out, cache7 = affine_forward(out, W6, b6)
    scores = out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(out, y)
    for i in np.arange(0,7):
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(i)])))
    dx, grads['W6'], grads['b6'] = affine_backward(dout, cache7)
    dx = dropout_backward(dx, do_cache)
    dx, grads['W5'], grads['b5'], grads['gamma5'], grads['beta5'] = affine_batnorm_relu_backward(dx, cache5)
    dx, grads['W4'], grads['b4'], grads['gamma4'], grads['beta4'] = conv_spabachnorm_relu_pool_backward(dx, cache4)
    dx, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = conv_spabachnorm_relu_pool_backward(dx, cache3)
    dx, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2']= conv_spabachnorm_relu_pool_backward(dx, cache2)

    dx, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1']= conv_spabachnorm_relu_backward(dx, cache1)
    dx, grads['W0'], grads['b0'], grads['gamma0'], grads['beta0']= conv_spabachnorm_relu_backward(dx, cache0)

    # add reg grad
    grads['W6'] += self.reg*W6
    grads['W5'] += self.reg*W5
    grads['W4'] += self.reg*W4
    grads['W3'] += self.reg*W3
    grads['W2'] += self.reg*W2
    grads['W1'] += self.reg*W1
    grads['W0'] += self.reg*W0
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

