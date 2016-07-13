from __future__ import division
from keras import backend as K
from keras.layers.core import Layer
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Bidirectional(Layer):
    ''' Bidirectional wrapper for RNNs

    # Arguments:
        rnn: `Recurrent` object.
        merge_mode: Mode by which outputs of the forward and reverse RNNs will be combined. One of {sum, mul, concat, ave}

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
        self.reverse = pickle.loads(pickle.dumps(rnn))
        self.forward.name = 'forward_' + self.forward.name
        self.reverse.name = 'reverse_' + self.reverse.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward.initial_weights = weights[:nw//2]
            self.reverse.initial_weights = weights[nw//2:]
        self._cache_enabled = True
        self.stateful = rnn.stateful
        self.return_sequences = rnn.return_sequences
        super(Bidirectional, self).__init__()

    def get_weights(self):
        return self.forward.get_weights() + self.reverse.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward.set_weights(weights[:nw//2])
        self.reverse.set_weights(weights[nw//2:])

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        self.forward.cache_enabled = value
        self.reverse.cache_enabled = value

    def get_output_shape_for(self, input_shape):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward.get_output_shape_for(input_shape)
        elif self.merge_mode == 'concat':
            shape = list(self.forward.get_output_shape_for(input_shape))
            shape[-1] *= 2
            return tuple(shape)

    def call(self, X, mask=None):
        def reverse(x):
            if K.ndim(x) == 2:
                x = K.expand_dims(x, -1)
                rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
                rev = K.squeeze(rev, -1)
            else:
                rev = K.permute_dimensions(x, (1, 0, 2))[::-1]                
            return K.permute_dimensions(rev, (1, 0, 2))

        Y = self.forward.call(X, mask) # 0,0,0,1,3,6,10
        X_rev = reverse(X) # 4,3,2,1,0,0,0
        mask_rev = reverse(mask) if mask else None # 1,1,1,1,0,0,0
        Y_rev = self.reverse.call(X_rev, mask_rev) # 4,7,9,10,10,10,10

        #Fix allignment
        if self.return_sequences:
            Y_rev = reverse(Y_rev) # 10,10,10,10,9,7,4

        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev

    @property
    def input_shape(self):
        return self.forward.input_shape

    def get_input(self, train=False):
        return self.forward.get_input(train)

    @property
    def non_trainable_weights(self):
        return self.forward.non_trainable_weights + self.reverse.non_trainable_weights

    @property
    def trainable_weights(self):
        return self.forward.trainable_weights + self.reverse.trainable_weights

    @trainable_weights.setter
    def trainable_weights(self, weights):
        nw = len(weights)
        self.forward.trainable_weights = weights[:nw//2]
        self.reverse.trainable_weights = weights[nw//2:]

    @non_trainable_weights.setter
    def non_trainable_weights(self, weights):
        nw = len(weights)
        self.forward.non_trainable_weights = weights[:nw//2]
        self.reverse.non_trainable_weights = weights[nw//2:]

    def reset_states(self):
        self.forward.reset_states()
        self.reverse.reset_states()

    def build(self, input_shape):
        self.forward.build(input_shape)
        self.reverse.build(input_shape)
        super(Bidirectional, self).build(input_shape)

    def get_config(self):
        config = {
                  "name": self.__class__.__name__,
                  "rnn": self.forward.get_config(),
                  "merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
