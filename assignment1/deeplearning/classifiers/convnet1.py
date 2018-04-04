import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *
from deeplearning.classifiers.fc_net import *

class HaoConvNet1(object):
  """
  [conv-relu] - [conv-relu-pool] - affine - bathnorm - relu - affine - softmax
  parameters number: ~ 16x3x32x32 + 16x16*32*32 + 16x16x16xhidden_dim + hidden_dimx10
    when hidden_dim = 256, total paramters ~ 1.4 millions
  memory usage per picture: ~ 16x32x32 + 16x32x32 + 16x16x16 + hidden_dim + hidden_dim + 10
    when hidden_din = 256, total memory ~ 37K/pic
  """


  def __init__(self, input_dim=(3, 32, 32), num_filters=16, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    input_c, input_h, input_w = input_dim

    #conv layer
    #input size (N, C, H, W)
    #output size (N,number_filters, H, W)
    #(H_out, W_out) = (H, W) because of the padding and stride config in the loss function, 
    self.params['W1'] = np.random.normal(scale = weight_scale, size = num_filters*input_c*filter_size**2).reshape((num_filters, input_c, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    
    #conv layer
    #input size (N, number_filters, H, W)
    #output size (N, number_filters, H, W)
    #(H_out, W_out) = (H, W) because of the padding and stride config in the loss function, 
    self.params['W2'] = np.random.normal(scale = weight_scale, size = (num_filters**2)*(filter_size**2)).reshape((num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)

    #pooling layer
    #pool_param is defined in the loss function
    #input size (N, number_filters, H, W) 
    #output size (N, number_filters, H/2, W/2)

    #hidden affine layer
    #input size (N, number_filters, H/2, W/2)
    #output size (N, hidden_dim)
    self.params['W3'] = np.random.normal(scale = weight_scale, size = num_filters*(input_h/2)*(input_w/2)*hidden_dim).reshape((num_filters*(input_h/2)*(input_w/2),hidden_dim))
    self.params['b3'] = np.zeros(hidden_dim)
    
    #output affine layer
    #input size (N, hidden_dim)
    #output size (N, num_classes)
    self.params['W4'] = np.random.normal(scale = weight_scale, size = hidden_dim*num_classes).reshape((hidden_dim, num_classes))
    self.params['b4'] = np.zeros(num_classes)

    self.params['gamma'] = np.ones(hidden_dim)
    self.params['beta'] = np.zeros(hidden_dim)

    self.bn_params = []
    self.bn_params = [{'mode': 'train'}]

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


    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
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
    gamma = self.params['gamma']
    beta = self.params['beta']
    bn_param = self.bn_params[0]

    out1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
    out3, cache3 = affine_batnorm_relu_forward(out2, W3, b3, gamma, beta, bn_param)
    out4, cache4 = affine_forward(out3, W4, b4)
    scores = out4
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
    loss, dout = softmax_loss(out4, y)
    for i in np.arange(1,5):
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(i)])))
    dx, grads['W4'], grads['b4'] = affine_backward(dout, cache4)
    dx, grads['W3'], grads['b3'], grads['gamma'], grads['beta'] = affine_batnorm_relu_backward(dx, cache3)
    dx, grads['W2'], grads['b2'] = conv_relu_pool_backward(dx, cache2)
    dx, grads['W1'], grads['b1'] = conv_relu_backward(dx, cache1)

    # add reg grad
    grads['W4'] += self.reg*W4
    grads['W3'] += self.reg*W3
    grads['W2'] += self.reg*W2
    grads['W1'] += self.reg*W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

