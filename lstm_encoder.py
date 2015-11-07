# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, sharedX
from six.moves import range
from stateful_rnn import StatefulRNN

class LSTMEncoder(StatefulRNN):

    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1,
                 input_dim=None, input_length=None, hidden_state=None, batch_size=None, decoder=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.initial_weights = weights
        self.initial_state = hidden_state
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_length = input_length
        self.decoder = decoder
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LSTMEncoder, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]
        nw = len(self.initial_weights) if self.initial_weights is not None else 0
        if self.initial_state is not None:
            self.h = sharedX(self.initial_state[0])
            self.c = sharedX(self.initial_state[1])
            del self.initial_state
        elif self.batch_size is not None:
            self.h = shared_zeros((self.batch_size, self.output_dim))
            self.c = shared_zeros((self.batch_size, self.output_dim))                
        elif self.initial_weights is not None:
            if nw == len(self.params) + 2:
                self.h = sharedX(self.initial_weights[-1])
                self.c = sharedX(self.initial_weights[-2])
                nw -= 2
            else:
                raise Exception("Hidden state not provided in weights")
        else:
            raise Exception("One of the following arguments must be provided for stateful RNNs: hidden_state, batch_size, weights")
        self.state = [self.h, self.c]
        self.params += self.state
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights[:nw])
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o         
        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[self.h, self.c],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)
        self.updates = ((self.h, outputs[-1][0]),(self.c, outputs[-1][1]) )
        if decoder is not None:
           	decoder_updates = ((decoder.h, outputs[-1][0]),(decoder.c, outputs[-1][1]))
           	self.updates += decoder_updates
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
