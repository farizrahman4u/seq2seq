from keras import backend as K
from copy import deepcopy
from warnings import warn
from keras.layers.core import MaskedLayer

class Bidirectional(MaskedLayer):
    ''' Bidirectional wrapper for RNNs

    # Arguments:
        rnn: `Recurrent` object. 
        merge_mode: Mode by which outputs of the forward and reverse RNNs will be combined. One of {sum, mul, concat, ave}

    # TensorFlow warning
        Limited accuracy for stateful bidirectional rnns with input mask when using TensorFlow backend 
    
    # Examples:
    ```python
    model = Sequential()
    model.add(Bidirectional(LSTM(10, input_shape=(10, 20))))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train,], Y_train, batch_size=32, nb_epoch=20,
              validation_data=([X_test], Y_test))
    ```
    '''
    def __init__(self, rnn, merge_mode='concat', weights=None):

        self.forward = rnn
        self.reverse = deepcopy(rnn)
        if K._BACKEND != 'theano':
            self.revere.return_sequences = True # Required due to padding issues
            if rnn.stateful:
                warn('Stateful bidirectional RNNs with input mask are fully supported only when using theano backend')

        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward.initial_weights = weights[:nw/2]
            self.reverse.initial_weights = weights[nw/2:]
        self._cache_enabled = True
        self.stateful = rnn.stateful
        self.return_sequences = rnn.return_sequences
        if hasattr(rnn, '_input_shape'):
            self._input_shape = rnn.input_shape
        elif hasattr(rnn, 'previous') and rnn.previous:
            self.previous = rnn.previous

    def get_weights(self):
        return self.forward.get_weights() + self.reverse.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward.set_weights(weights[:nw/2])
        self.reverse.set_weights(weights[:nw/2])

    def set_previous(self, layer):
        self.previous = layer
        self.forward.set_previous(layer)
        self.reverse.set_previous(layer)
        self._input_shape = layer.output_shape

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        self.forward.cache_enabled = value
        self.reverse.cache_enabled = value

    @property
    def output_shape(self):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward.output_shape
        elif self.merge_mode == 'concat':
            shape = list(self.forward.output_shape)
            shape[-1] *= 2
            return tuple(shape)

    def get_output(self, train=False):
        X = self.get_input(train) # 0,0,0,1,2,3,4

        mask = self.get_input_mask(train) # 0,0,0,1,1,1,1

        # X_rev = reverse(X)
        X_rev = K.permute_dimensions(X, (1, 0, 2))
        X_rev = X_rev[::-1]
        X_rev = K.permute_dimensions(X_rev, (1, 0, 2)) # 4,3,2,1,0,0,0

        Y = self.forward(X, mask) # 0,0,0,1,3,6,10

        Y_rev = None

        if mask:

            if K._BACKEND == 'theano':
                #convert right padding to left padding by rolling
                shifts = K.sum(mask, axis=1)
                import theano
                X_rev, _ = theano.scan(lambda x, i: theano.tensor.roll(x, -i, 0),
                             sequences=[X_rev, shifts]) # 0,0,0,4,3,2,1

                #Get reverse output
                Y_rev = self.reverse(X_rev, mask) # 0,0,0,4,7,9,10 or just 10 if return_sequences = False

                if self.return_sequences:

                    #Fix allignment : 
                    # When return_sequence = True, outputs corresponding to the same input should be merged.

                    # Reverse Y_rev.
                    # Note : On reversing left padding will be converted to right padding.
                    Y_rev = K.permute_dimensions((1, 0, 2))
                    Y_rev = Y_rev[::-1]
                    Y_rev = K.permute_dimensions((1, 0, 2)) # 10,9,7,4,0,0,0

                    #Convert right padding back to to left padding              
                    Y_rev, _ = theano.scan(lambda x, i: theano.tensor.roll(x, -i, 0),
                             sequences=[Y_rev, shifts]) # 0,0,0,10,9,7,4
            else:

                import tensorflow as tf

                # mask_rev = reverse(mask)
                mask_rev = K.permute_dimensions(mask, (1, 0))
                mask_rev = mask_rev[::-1]
                mask_rev = K.permute_dimensions(mask_rev, (1, 0)) # 1,1,1,1,0,0,0

                # X_rev = 4,3,2,1,0,0,0
                # Get reverse output:
                Y_rev = self.reverse(X_rev, mask_rev) # 4,7,9,10,g,g,g  (g = Garbage value)

                # Reverse Y_rev
                Y_rev = K.permute_dimensions(Y_rev, (1, 0, 2))
                Y_rev = Y_rev[::-1]
                Y_rev = K.permute_dimensions(Y_rev, (1, 0, 2)) # g,g,g,10,9,7,4              

                # Trim off garbage values
                [garbage, Y_rev] = tf.dynamic_partition(Y_rev, mask, 2) # [g,g,g] [10,9,7,4]

                if self.return_sequences:
                    #pad left
                    zeros = K.zeros_like(garbage) # 0,0,0
                    Y_rev = K.concatenate([zeros, Y_rev], axis=1) # 0,0,0,10,9,7,4
                else:
                    Y_rev = Y_rev[:,0] # 10

        else:

            self.reverse.return_sequences = self.return_sequences
            Y_rev = self.reverse(X_rev)
            if self.return_sequences:
                Y_rev = K.permute_dimensions(Y_rev, (1, 0, 2))
                Y_rev = Y_rev[::-1]
                Y_rev = K.permute_dimensions(Y_rev, (1, 0, 2))
            if K._BACKEND != 'theano':
                self.revere.return_sequences = True        

        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev

    def get_output_mask(self, train=False):
        if self.forward.return_sequences:
            return self.get_input_mask(train)
        else:
            return None

    @property
    def input_shape(self):
        return self.forward.input_shape

    def get_input(self, train=False):
        return self.forward.get_input(train)

    @property
    def params(self):
        return self.forward.get_params()[0] + self.reverse.get_params()[0]

    @property
    def regularizers(self):
        return self.forward.get_params()[1] + self.reverse.get_params()[1] 

    @property
    def constraints(self):
        return self.forward.get_params()[2] + self.reverse.get_params()[2]

    @property
    def updates(self):
        return self.forward.get_params()[3] + self.reverse.get_params()[3]

    def reset_states(self):
        self.forward.reset_states()
        self.reverse.reset_states()

    def build(self):
        if not hasattr(self.forward, '_input_shape'):
            if hasattr(self, '_input_shape'):
                self.forward._input_shape = self._input_shape
                self.reverse._input_shape = self._input_shape
                self.forward.previous = self.previous
                self.reverse.previous = self.previous
                self.forward.params = []
                self.reverse.params = []
                self.forward.build()
                self.reverse.build()

    def get_config(self):
        config = {"rnn": self.forward.get_config(),
                  "merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
