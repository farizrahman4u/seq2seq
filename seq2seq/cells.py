import recurrentshop
from recurrentshop.cells import ExtendedRNNCell
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras import backend as K


class LSTMDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if not hidden_dim:
            self.hidden_dim = kwargs['units ']
        self.hidden_dim = hidden_dim
        super(ExtendedRNNCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim
        output_shape = (input_shape[0], output_dim)

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)

        W1 = Dense(output_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   use_bias=False)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,)

        z = add([W1(x), U(h_tm1)])
        z0, z1, z2, z3 = get_slices(z, 4)
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = add([multiply([f, c_tm1]), multiply([i, self.activation(z2)])])
        o = self.recurrent_activation(z3)
        h = multiply([o, self.activation(c)])
        y = self.activation(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])


class AttentionDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if not hidden_dim:
            self.hidden_dim = kwargs['units']
        self.hidden_dim = hidden_dim
        super(AttentionDecoderCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        output_shape = (input_shape[0], output_dim)
        hidden_dim = self.hidden_dim
        input_length = input_shape[1]

        x = Input(batch_shape=input_shape)
        H = x
        h_tm1 = Input(batch_shape=output_shape)
        c_tm1 = Input(batch_shape=output_shape)

        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W3 = Dense(1,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer)

        C = Lambda(lambda x: K.repeat(x, input_length))(c_tm1)
        _HC = concatenate([H, C])
        _HC = Lambda(lambda x: K.reshape(x, (-1, input_dim + self.hidden_dim)))(_HC)
        
        energy = W3(_HC)
        energy = Lambda(lambda x: K.reshape(x, (-1, input_length)))(energy)
        energy = Activation('softmax')(energy)

        x = Lambda(lambda x: K.batch_dot(energy, x, axes=(1, 1)))(H)
        z = add([W1(x), U(h_tm1)])
        z0, z1, z2, z3 = get_slices(z, 4)
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z0)
        c = add([multiply([f, c_tm1]), multiply(i, self.activation(z2))])
        o = self.recurrent_activation(z3)
        h = multiply([o, self.activation(c)])
        y = self.activation(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])
