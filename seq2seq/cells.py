import recurrentshop
from recurrentshop.cells import *
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras import backend as K


class LSTMDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        super(LSTMDecoderCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4,
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
        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)
        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])


class AttentionDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        self.input_ndim = 3
        super(AttentionDecoderCell, self).__init__(**kwargs)


    def build_model(self, input_shape):
        
        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        
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

        C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
        _xC = concatenate([x, C])
        _xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)

        alpha = W3(_xC)
        alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
        alpha = Activation('softmax')(alpha)

        _x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])

        z = add([W1(_x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)

        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])
