# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras import backend as K
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras import activations, initializations
import numpy as np


def get_state_transfer_rnn(RNN):
    '''Converts a given Recurrent sub class (e.g, LSTM, GRU) to its state transferable version.
    A state transfer RNN can transfer its hidden state to another one of the same type and compatible dimensions.
    '''

    class StateTransferRNN(RNN):

        def __init__(self, state_input=True, **kwargs):
            self.state_outputs = []
            self.state_input = state_input
            super(StateTransferRNN, self).__init__(**kwargs)

        def reset_states(self):
            stateful = self.stateful
            self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
            if self.stateful:
                super(StateTransferRNN, self).reset_states()
            self.stateful = stateful

        def build(self,input_shape):
            stateful = self.stateful
            self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
            super(StateTransferRNN, self).build(input_shape)
            self.stateful = stateful

        def broadcast_state(self, rnns):
            if type(rnns) not in [list, tuple]:
                rnns = [rnns]
            rnns = list(set(rnns) - set(self.state_outputs))
            self.state_outputs += rnns
            for rnn in rnns:
                rnn.state_input = self
            for rnn in rnns:
                if not hasattr(rnn, 'updates'):
                    rnn.updates = []
                for i in range(len(rnn.states)):
                    rnn.updates.append((rnn.states[i], self.states_to_transfer[i]))

        def call(self, x, mask=None):
            input_shape = self.input_spec[0].shape
            if K._BACKEND == 'tensorflow':
                if not input_shape[1]:
                    raise Exception('When using TensorFlow, you should define '
                                    'explicitly the number of timesteps of '
                                    'your sequences.\n'
                                    'If your first layer is an Embedding, '
                                    'make sure to pass it an "input_length" '
                                    'argument. Otherwise, make sure '
                                    'the first layer has '
                                    'an "input_shape" or "batch_input_shape" '
                                    'argument, including the time axis. '
                                    'Found input shape at layer ' + self.name +
                                    ': ' + str(input_shape))

            if self.stateful or self.state_input or len(self.state_outputs) > 0:
                initial_states = self.states
            else:
                initial_states = self.get_initial_states(x)

            constants = self.get_constants(x)
            preprocessed_input = self.preprocess_input(x)
            last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                                 initial_states,
                                                 go_backwards=self.go_backwards,
                                                 mask=mask,
                                                 constants=constants,
                                                 unroll=self.unroll,
                                                 input_length=input_shape[1])

            n = len(states)
            if self.stateful and not self.state_input:
                self.updates = []
                for i in range(n):
                    self.updates.append((self.states[i], states[i]))
            self.states_to_transfer = states
            if self.return_sequences:
                return outputs
            else:
                return last_output
    return StateTransferRNN


StateTransferSimpleRNN = get_state_transfer_rnn(SimpleRNN)
StateTransferGRU = get_state_transfer_rnn(GRU)
StateTransferLSTM = get_state_transfer_rnn(LSTM)
