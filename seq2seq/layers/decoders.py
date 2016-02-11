# -*- coding: utf-8 -*-

from __future__ import absolute_import
from keras import backend as K
from keras import activations, initializations
from seq2seq.layers.state_transfer_lstm import StateTransferLSTM
import theano
import numpy as np

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


class LSTMDecoder(StateTransferLSTM):
    '''
    A basic LSTM decoder. Similar to [1].
    The output of at each timestep is the input to the next timestep.
    The input to the first timestep is the context vector from the encoder.

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    In addition, the hidden state of the encoder is usually used to initialize the hidden
    state of the decoder. Checkout models.py to see how its done.
    '''
    input_ndim = 2

    def __init__(self, output_length, hidden_dim=None, **kwargs):

        self.output_length = output_length
        self.hidden_dim = hidden_dim
        input_dim = None
        if 'input_dim' in kwargs:
            kwargs['output_dim'] = input_dim
        if 'input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['input_shape'][-1]
        elif 'batch_input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['batch_input_shape'][-1]
        super(LSTMDecoder, self).__init__(**kwargs)
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []

    def set_previous(self, layer, connection_map={}):
        '''Connect a layer to its parent in the computational graph.
        '''
        self.previous = layer
        self.build()

    def build(self):
        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.output_dim = dim
        self.input = K.placeholder(input_shape)
        if not self.hidden_dim:
            self.hidden_dim = dim
        hdim = self.hidden_dim
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (hidden_dim)
            self.states = [None, None]

        self.W_i = self.init((dim, hdim))
        self.U_i = self.inner_init((hdim, hdim))
        self.b_i = K.zeros((hdim))

        self.W_f = self.init((dim, hdim))
        self.U_f = self.inner_init((hdim, hdim))
        self.b_f = self.forget_bias_init((hdim))

        self.W_c = self.init((dim, hdim))
        self.U_c = self.inner_init((hdim, hdim))
        self.b_c = K.zeros((hdim))

        self.W_o = self.init((dim, hdim))
        self.U_o = self.inner_init((hdim, hdim))
        self.b_o = K.zeros((hdim))

        self.W_x = self.init((hdim, dim))
        self.b_x = K.zeros((dim))

        self.trainable_weights = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_x, self.b_x
        ]

    def reset_states(self):
        assert self.stateful or self.state_input or len(self.state_outputs > 0), 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.hidden_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.hidden_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.hidden_dim)),
                           K.zeros((input_shape[0], self.hidden_dim))]

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, hidden_dim)
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def _step(self,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, b_i, b_f, b_c, b_o, b_x):

        xi_t = K.dot(x_tm1, w_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t

    def get_output(self, train=False):
        x_t = self.get_input(train)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x_t)
        [outputs, hidden_states, cell_states], updates = theano.scan(
            self._step,
            n_steps=self.output_length,
            outputs_info=[x_t] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                           self.W_i, self.W_f, self.W_c, self.W_o,
                           self.W_x, self.b_i, self.b_f, self.b_c,
                           self.b_o, self.b_x])

        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    @property
    def output_shape(self):
        shape = list(super(LSTMDecoder, self).output_shape)
        shape[1] = self.output_length
        return tuple(shape)

    def get_config(self):
        config = {'name': self.__class__.__name__, 
        'hidden_dim': self.hidden_dim,
        'output_length': self.output_length}
        base_config = super(LSTMDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMDecoder2(LSTMDecoder):
    '''
    This decoder is similar to the first one, except that at every timestep the decoder gets
    a peek at the context vector.
    Similar to [2].

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector 
        from the encoder.

    '''
    def build(self):
        super(LSTMDecoder2, self).build()
        dim = self.input_dim
        hdim = self.hidden_dim
        self.V_i = self.init((dim, hdim))
        self.V_f = self.init((dim, hdim))
        self.V_c = self.init((dim, hdim))
        self.V_o = self.init((dim, hdim))
        self.trainable_weights += [self.V_i, self.V_c, self.V_f, self.V_o]

    def _step(self,
              x_tm1,
              h_tm1, c_tm1, v,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x):

        #Inputs = output from previous time step, vector from encoder
        xi_t = K.dot(x_tm1, w_i) + K.dot(v, v_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + K.dot(v, v_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + K.dot(v, v_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + K.dot(v, v_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t

    def get_output(self, train=False):
        v = self.get_input(train)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(v)        
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self._step,
            n_steps = self.output_length,
            outputs_info=[v] + initial_states,
            non_sequences=[v, self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c, 
                          self.b_o, self.b_x])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LSTMDecoder2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionDecoder(LSTMDecoder2):
    '''
    This is an attention decoder based on [3].
    Unlike the other decoders, AttentionDecoder requires the encoder to return
    a sequence of hidden states, instead of just the final context vector.
    Or in Keras language, while using AttentionDecoder, the encoder should have
    return_sequences = True.
    Also, the encoder should be a bidirectional RNN for best results.

    Working:

    A sequence of vectors X = {x0, x1, x2,....xm-1}, where m = input_length is input
    to the encoder.

    The encoder outputs a hidden state at each timestep H = {h0, h1, h2,....hm-1}

    The decoder uses H to generate a sequence of vectors Y = {y0, y1, y2,....yn-1}, 
    where n = output_length

    Decoder equations:

        Note: hk means H(k).

        y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
        and v (called the context vector) is a weighted sum over H:

        v(i) =  sigma(j = 0 to m-1)  alpha[i, j] * hj

        The weight alpha(i, j) for each hj is computed as follows:
        energy = a(s(i-1), hj)       
        alhpa = softmax(energy)
        Where a is a feed forward network.
    '''

    input_ndim = 3

    def build(self):
        super(AttentionDecoder, self).build()
        dim = self.input_dim
        hdim = self.hidden_dim
        self.input_length = self.input_shape[-2]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
        self.W_h = self.init((dim, hdim))
        self.b_h = K.zeros((hdim, ))
        self.W_a = self.init((hdim, 1))
        self.b_a = K.zeros((1,))
        self.trainable_weights += [self.W_a, self.b_a, self.W_h, self.b_h]

    def _step(self,
              x_tm1,
              h_tm1, c_tm1, H,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x, w_a, v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x, b_a):

        s_tm1 = K.repeat(c_tm1, self.input_length)
        e = H + s_tm1
        def a(x, states):
            output = K.dot(x, w_a) + b_a
            return output, []
        _, energy, _ = K.rnn(a, e, [], mask=None)
        energy = activations.get('linear')(energy)
        energy = K.permute_dimensions(energy, (2, 0, 1))
        energy = energy[0]
        alpha = K.softmax(energy)
        alpha = K.repeat(alpha, self.hidden_dim)
        alpha = K.permute_dimensions(alpha, (0, 2 , 1))
        weighted_H = H * alpha
        
        v = K.sum(weighted_H, axis=1)

        xi_t = K.dot(x_tm1, w_i) + K.dot(v, v_i) + b_i
        xf_t = K.dot(x_tm1, w_f) + K.dot(v, v_f) + b_f
        xc_t = K.dot(x_tm1, w_c) + K.dot(v, v_c) + b_c
        xo_t = K.dot(x_tm1, w_o) + K.dot(v, v_o) + b_o

        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)

        x_t = K.dot(h_t, w_x) + b_x
        return x_t, h_t, c_t
       
    def get_output(self, train=False):
        H = self.get_input(train)
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        def reshape(x, states):
            h = K.dot(x, self.W_h) + self.b_h
            return h, []
        _, H, _ = K.rnn(reshape, H, [], mask=None)
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self._step,
            n_steps = self.output_length,
            outputs_info=[X] + initial_states,
            non_sequences=[H, self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.W_a, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c, 
                          self.b_o, self.b_x, self.b_a])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
