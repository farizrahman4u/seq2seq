from recurrentshop import RNNCell, LSTMCell, weight
from keras import backend as K
import numpy as np


class LSTMDecoderCell(LSTMCell):

	def __init__(self, hidden_dim=None, **kwargs):
		if not hidden_dim:
			self.hidden_dim = kwargs['output_dim']
		self.hidden_dim = hidden_dim
		super(LSTMDecoderCell, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		W1 = weight((input_dim, 4 * self.hidden_dim,), init=self.init, regularizer=self.W_regularizer, name='{}_W1'.format(self.name))
		W2 = weight((self.hidden_dim, self.output_dim), init=self.init, regularizer=self.W_regularizer, name='{}_W2'.format(self.name))
		U = weight((self.hidden_dim, 4 * self.hidden_dim,), init=self.inner_init, regularizer=self.U_regularizer, name='{}_U'.format(self.name))
		b1 = np.concatenate([np.zeros(self.hidden_dim), K.get_value(self.forget_bias_init((self.hidden_dim,))), np.zeros(2 * self.hidden_dim)])
		b1 = weight(b1, regularizer=self.b_regularizer, name='{}_b1'.format(self.name))
		b2 = weight((self.output_dim,), init='zero', regularizer=self.b_regularizer, name='{}_b2'.format(self.name))
		h = (-1, self.hidden_dim)
		c = (-1, self.hidden_dim)

		def step(x, states, weights):
			h_tm1, c_tm1 = states
			W1, W2, U, b1, b2 = weights
			z = K.dot(x, W1) + K.dot(h_tm1, U) + b1
			z0 = z[:, :self.hidden_dim]
			z1 = z[:, self.hidden_dim: 2 * self.hidden_dim]
			z2 = z[:, 2 * self.hidden_dim: 3 * self.hidden_dim]
			z3 = z[:, 3 * self.hidden_dim:]
			i = self.inner_activation(z0)
			f = self.inner_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.inner_activation(z3)
			h = o * self.activation(c)
			y = self.activation(K.dot(h, W2) + b2)
			return y, [h, c]

		self.step = step
		self.states = [h, c]
		self.weights = [W1, W2, U, b1, b2]
		super(LSTMCell, self).build(input_shape)

	def get_config(self):
		config = {'hidden_dim': self.hidden_dim}
		base_config = super(LSTMDecoderCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class AttentionDecoderCell(LSTMCell):

	def __init__(self, hidden_dim=None, **kwargs):
		if not hidden_dim:
			self.hidden_dim = kwargs['output_dim']
		self.hidden_dim = hidden_dim
		super(AttentionDecoderCell, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		input_length = input_shape[1]
		W1 = weight((input_dim, 4 * self.hidden_dim,), init=self.init, regularizer=self.W_regularizer, name='{}_W1'.format(self.name))
		W2 = weight((self.hidden_dim, self.output_dim), init=self.init, regularizer=self.W_regularizer, name='{}_W2'.format(self.name))
		W3 = weight((self.hidden_dim + input_dim, 1), init=self.init, regularizer=self.W_regularizer, name='{}_W3'.format(self.name))
		U = weight((self.hidden_dim, 4 * self.hidden_dim,), init=self.inner_init, regularizer=self.U_regularizer, name='{}_U'.format(self.name))
		b1 = np.concatenate([np.zeros(self.hidden_dim), K.get_value(self.forget_bias_init((self.hidden_dim,))), np.zeros(2 * self.hidden_dim)])
		b1 = weight(b1, regularizer=self.b_regularizer, name='{}_b1'.format(self.name))
		b2 = weight((self.output_dim,), init='zero', regularizer=self.b_regularizer, name='{}_b2'.format(self.name))
		b3 = weight((1,), init='zero', regularizer=self.b_regularizer, name='{}_b3'.format(self.name))
		h = (-1, self.hidden_dim)
		c = (-1, self.hidden_dim)

		def step(x, states, weights):
			H = x
			h_tm1, c_tm1 = states
			W1, W2, W3, U, b1, b2, b3 = weights
			input_length = K.shape(x)[1]
			C = K.repeat(c_tm1, input_length)
			_HC = K.concatenate([H, C])
			_HC = K.reshape(_HC, (-1, input_dim + self.hidden_dim))
			energy = K.dot(_HC, W3) + b3
			energy = K.reshape(energy, (-1, input_length))
			energy = K.softmax(energy)
			x = K.batch_dot(energy, H, axes=(1, 1))
			z = K.dot(x, W1) + K.dot(h_tm1, U) + b1
			z0 = z[:, :self.hidden_dim]
			z1 = z[:, self.hidden_dim: 2 * self.hidden_dim]
			z2 = z[:, 2 * self.hidden_dim: 3 * self.hidden_dim]
			z3 = z[:, 3 * self.hidden_dim:]
			i = self.inner_activation(z0)
			f = self.inner_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.inner_activation(z3)
			h = o * self.activation(c)
			y = self.activation(K.dot(h, W2) + b2)
			return y, [h, c]

		self.step = step
		self.weights = [W1, W2, W3, U, b1, b2, b3]
		self.states = [h, c]

		super(RNNCell, self).build(input_shape)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_config(self):
		config = {'hidden_dim': self.hidden_dim}
		base_config = super(AttentionDecoderCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
