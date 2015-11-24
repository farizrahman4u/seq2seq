#Bidirectional Deep LSTM layer
#API similar to that of DeepLSTM
#Reads input in forward and backward directions
#For depth > 1, hidden state is  propogated throughout the LSTM stack

from keras.layers.core import TimeDistributedDense, Dropout, Merge

from seq2seq.deep_lstm import DeepLSTM
from seq2seq.stateful_container import StatefulContainer


class BidirectionalLSTM(StatefulContainer):

    def __init__(self, input_dim, input_length, output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1,
                  hidden_state=None, batch_size=None, depth=1, remember_state=False,
                 inner_return_sequences=True, return_sequences=True):

        if not weights:
            weights = [None]*5#No weights for merge layer
        if not hidden_state:
            hidden_state = [None]*6
        super(StatefulContainer, self).__init__()

        forward = DeepLSTM(input_dim=input_dim*2, output_dim=output_dim,
                        input_length=input_length,
                        weights=weights[2], hidden_state=hidden_state[2],
                        batch_size=batch_size,depth=depth, remember_state=remember_state,
                        inner_return_sequences=inner_return_sequences,
                        return_sequences=return_sequences, init='glorot_uniform', 
                        inner_init='orthogonal', forget_bias_init='one',
                        activation='tanh', inner_activation='hard_sigmoid')

        reverse = DeepLSTM(input_dim=input_dim*2, output_dim=output_dim,
                        input_length=input_length,
                        weights=weights[3], hidden_state=hidden_state[3],
                        batch_size=batch_size,depth=depth, remember_state=remember_state,
                        inner_return_sequences=inner_return_sequences,
                        return_sequences=return_sequences, init='glorot_uniform', 
                        inner_init='orthogonal', forget_bias_init='one',
                        activation='tanh', inner_activation='hard_sigmoid', go_backwards=True)

        #A common input to both forward and reverse LSTMs
        #This layer learns a direction invariant representation of your input data
        self.add(TimeDistributedDense(input_dim=input_dim, output_dim=input_dim*2, 
                                input_length=input_length))

        if weights[0]:
            self.layers[0].set_weights(weights[0])

        self.add(Dropout(0.7))
        self.add(forward)
        self.add(reverse)
        reverse.set_previous(forward.layers[0].previous)#Woah!
        merge = Merge([forward, reverse], mode='concat', concat_axis=-1)
        layers = self.layers[:2]
        for l in layers:
            params, regs, consts, updates = l.get_params()
            merge.regularizers += regs
            merge.updates += updates
            for p, c in zip(params, consts):
                if p not in merge.params:
                    merge.params.append(p)
                    merge.constraints.append(c)
        self.add(merge)
        if return_sequences:
            self.add(TimeDistributedDense(output_dim))
        else:
            self.add(Dense(output_dim))

    @property
    def params(self):
        params = []
        layers = self.layers[:]
        layers.pop(4)
        for l in layers:
            if l.trainable:
                params += l.get_params()[0]
        return params

    @property
    def regularizers(self):
        regularizers = []
        layers = self.layers[:]
        layers.pop(4)
        for l in layers:
            if l.trainable:
                regularizers += l.get_params()[1]
        return regularizers

    @property
    def constraints(self):
        constraints = []
        layers = self.layers[:]
        layers.pop(4)
        for l in layers:
            if l.trainable:
                constraints += l.get_params()[2]
        return constraints

    @property
    def updates(self):
        updates = []
        layers = self.layers[:]
        layers.pop(4)
        for l in layers:
            if l.trainable:
                updates += l.get_params()[3]
        return updates

    def get_weights(self):
        layers = self.layers[:]
        layers.pop(4)
        return [l.get_weights() for l in layers]

    def set_weights(self, weights):
        layers = self.layers[:]
        layers.pop(4)
        if len(layers) != len(weights):
            raise Exception("Exactly " + str(len(layers)) + " weight arrays required " + 
                str(len(weights)) + " given")
        for l, w in zip(layers, weights):
            l.set_weights(w)
