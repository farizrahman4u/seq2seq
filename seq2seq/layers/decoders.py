# -*- coding: utf-8 -*-

from __future__ import absolute_import
from keras import backend as K
from keras import activations, initializations
from keras.engine import InputSpec
from keras.layers.recurrent import time_distributed_dense
from seq2seq.layers.state_transfer_rnn import StateTransferLSTM
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
        elif 'output_dim' not in kwargs:
            kwargs['output_dim'] = None
        super(LSTMDecoder, self).__init__(**kwargs)
        self.return_sequences = True
        self.updates = []
        self.consume_less = 'mem'

    def build(self, input_shape):
        input_shape = list(input_shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        if not self.hidden_dim:
            self.hidden_dim = input_shape[-1]
        output_dim = input_shape[-1]
        self.output_dim = self.hidden_dim
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(LSTMDecoder, self).build(input_shape)
        self.output_dim = output_dim
        self.initial_weights = initial_weights
        self.W_y = self.init((self.hidden_dim, self.output_dim), name='{}_W_y'.format(self.name))
        self.b_y = K.zeros((self.output_dim), name='{}_b_y'.format(self.name))
        self.trainable_weights += [self.W_y, self.b_y]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=tuple(input_shape))]

    def get_constants(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        consts = super(LSTMDecoder, self).get_constants(x)
        self.output_dim = output_dim
        return consts

    def reset_states(self):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        super(LSTMDecoder, self).reset_states()
        self.output_dim = output_dim

    def get_initial_states(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        initial_states = super(LSTMDecoder, self).get_initial_states(x)
        self.output_dim = output_dim
        return initial_states

    def step(self, x, states):
        assert len(states) == 5, len(states)
        states = list(states)
        y_tm1 = states.pop(2)
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        h_t, new_states = super(LSTMDecoder, self).step(y_tm1, states)
        self.output_dim = output_dim
        y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
        new_states += [y_t]
        return y_t, new_states

    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)
        input_shape = list(self.input_spec[0].shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        self.input_spec = [InputSpec(shape=tuple(input_shape))]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states[:]
        else:
            initial_states = self.get_initial_states(X)
        constants = self.get_constants(X)
        y_0 = K.permute_dimensions(X, (1, 0, 2))[0, :, :]
        initial_states += [y_0]
        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        self.states_to_transfer = states
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=input_shape)]
        return outputs

    def assert_input_compatibility(self, x):
        shape = x._keras_shape
        assert K.ndim(x) == 2, "LSTMDecoder requires 2D  input, not " + str(K.ndim(x)) + "D."
        assert shape[-1] == self.output_dim or not self.output_dim, "output_dim of LSTMDecoder should be same as the last dimension in the input shape. output_dim = "+ str(self.output_dim) + ", got tensor with shape : " + str(shape) + "."

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        return tuple(output_shape)

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
    def build(self, input_shape):
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(LSTMDecoder2, self).build(input_shape)
        self.initial_weights = initial_weights
        dim = self.input_spec[0].shape[-1]
        self.W_x = self.init((dim, dim), name='{}_W_x'.format(self.name))
        self.b_x = K.zeros((dim,), name='{}_b_x'.format(self.name))
        self.trainable_weights += [self.W_x, self.b_x]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        assert len(states) == 5, len(states)
        states = list(states)
        y_tm1 = states.pop(2)
        v = self.activation(K.dot(x, self.W_x) + self.b_x)
        y_tm1 += v
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        h_t, new_states = super(LSTMDecoder, self).step(y_tm1, states)
        self.output_dim = output_dim
        y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
        new_states += [y_t]
        return y_t, new_states

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

    def build(self, input_shape):
        self.input_length = input_shape[1]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(AttentionDecoder, self).build(input_shape[:1] + input_shape[2:])
        self.initial_weights = initial_weights
        dim = self.input_dim
        hdim = self.hidden_dim
        self.W_h = self.init((hdim, dim), name='{}_W_h'.format(self.name))
        self.b_h = K.zeros((dim, ), name='{}_b_h'.format(self.name))
        self.W_a = self.init((dim, 1), name='{}_W_a'.format(self.name))
        self.b_a = K.zeros((1,), name='{}_b_a'.format(self.name))
        self.trainable_weights += [self.W_a, self.b_a, self.W_h, self.b_h]
        if self.initial_weights is not None:
            self.set_weights(self.inital_weights)
            del self.initial_weights

    def step(self, x, states):
        h_tm1, c_tm1, y_tm1, B, U, H = states
        s = K.dot(c_tm1, self.W_h) + self.b_h
        s = K.repeat(s, self.input_length)
        energy = time_distributed_dense(s + H, self.W_a, self.b_a)
        energy = K.squeeze(energy, 2)
        alpha = K.softmax(energy)
        alpha = K.repeat(alpha, self.input_dim)
        alpha = K.permute_dimensions(alpha, (0, 2, 1))
        weighted_H = H * alpha
        v = K.sum(weighted_H, axis=1)
        y, new_states = super(AttentionDecoder, self).step(v, states[:-1])
        return y, new_states

    def call(self, x, mask=None):
        H = x
        x = K.permute_dimensions(H, (1, 0, 2))[-1, :, :]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states[:]
        else:
            initial_states = self.get_initial_states(H)
        constants = self.get_constants(H) + [H]
        y_0 = x
        x = K.repeat(x, self.output_length)
        initial_states += [y_0]
        last_output, outputs, states = K.rnn(self.step, x,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        self.states_to_transfer = states
        return outputs

    def assert_input_compatibility(self, x):
        shape = x._keras_shape
        assert K.ndim(x) == 3, "AttentionDecoder requires 3D  input, not " + str(K.ndim(x)) + "D."
        assert shape[-1] == self.output_dim or not self.output_dim, "output_dim of AttentionDecoder should be same as the last dimension in the input shape. output_dim = "+ str(self.output_dim) + ", got tensor with shape : " + str(shape) + "."

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = input_shape[:1] + [self.output_length] + input_shape[2:]
        return tuple(output_shape)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
