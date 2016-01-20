# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras import activations, initializations
import numpy as np

class StateTransferLSTM(LSTM):

    def __init__(self, state_input=True, **kwargs):
        self.state_outputs = []
        self.state_input = state_input
        super(StateTransferLSTM, self).__init__(**kwargs)

    def  build(self):
        stateful = self.stateful
        self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
        if hasattr(self, 'states'):
            del self.states
        super(StateTransferLSTM, self).build()
        self.stateful = stateful

    def broadcast_state(self, rnns):
        if type(rnns) not in [list, tuple]:
            rnns = [rnns]
        self.state_outputs += rnns
        for rnn in rnns:
            rnn.state_input = self

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)

        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = K.rnn(self.step, X, initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask)
        n = len(states)
        if self.stateful and not self.state_input:
            self.updates = []
            self.updates = []
            for i in range(n):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(n):
                o.updates.append((o.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def set_input_shape(self, shape):

        self._input_shape = shape
        self.build()
