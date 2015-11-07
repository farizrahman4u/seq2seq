import keras
from keras.models import Sequential
from keras.layers.core import Dense

from lstm_encoder import LSTMEncoder
from lstm_decoder import LSTMDecoder

class Seq2seq(Sequential):
	def __init__(self, output_dim, hidden_dim,output_length, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1,
                 input_dim=None, input_length=None, hidden_state=None, batch_size=None, depth=1
                 ):

		if not type(depth) == list:
			depth = [depth, depth]
		n_lstms = sum(depth)

		if weights is None:
			weights = [None] * (n_lstms + 1)

		if hidden_state is None:
			hidden_state = [None] * n_lstms

		encoder_weight_index = depth[0] - 1
		decoder_weight_index = depth[0] + 1

		encoder_state_index = depth[0] - 1
		decoder_state_index = depth[0]

		decoder = LSTMDecoder(dim=output_dim, hidden_dim=hidden_dim, output_length=output_length,
							  init=init,inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[decoder_weight_index],
							  truncate_gradient = truncate_gradient, 
							  hidden_state=hidden_state[decoder_state_index], batch_size=batch_size)

		encoder = LSTMEncoder(input_dim=input_dim, output_dim=hidden_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[encoder_weight_index],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[encoder_state_index], batch_size=batch_size, decoder=decoder)

		left_deep = [LSTMEncoder(input_dim=input_dim, output_dim=input_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[i],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[i], batch_size=batch_size, return_sequences=True)
					for i in range(depth[0]-1)]


		right_deep = [LSTMEncoder(input_dim=output_dim, output_dim=output_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[decoder_weight_index + 1 + i],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[decoder_state_index + 1 + i], batch_size=batch_size, return_sequences=True)
					for i in range(depth[1]-1)]

		dense = Dense(input_dim=hidden_dim, output_dim=output_dim)

		if weights[1] is not None:
			dense.set_weights(weights[1])
		super(Seq2seq, self).__init__()
		for l in left_deep:
			self.add(l)
		self.add(encoder)
		self.add(dense)
		self.add(decoder)
		for l in right_deep:
			self.add(l)
		self.encoder = encoder
		self.dense = dense
		self.decoder = decoder

	def get_hidden_state(self):
		state = []
		for i in range(len(self.layers)):
			if hasattr(self.layers[i], 'state'):
				state.append(self.layers[i].get_hidden_state())
		return state	

def set_hidden_state(self, state):
	for i in range(len(self.layers)):
		if hasattr(self.layers[i], 'state'):
			self.layers[i].set_hidden_state(state[i])

def get_weights(self):
	return [l.get_weights() for l in self.layers]

def set_weights(self, weights):
	if len(self.layers) != len(weights):
		raise Exception("Exactly " + str(len(self.layers)) + " weight arrays required " + 
			str(len(weights)) + " given")
	for l,w in zip(l,weights):
		l.set_weights(w)
