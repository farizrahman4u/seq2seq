import keras
from keras.models import Sequential
from keras.layers.core import Dense

from lstm_encoder import LSTMEncoder
from lstm_decoder import LSTMDecoder

class Seq2seq(Sequential):
	def __init__(self, output_dim, hidden_dim,output_length, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                 weights=[None,None], truncate_gradient=-1,
                 input_dim=None, input_length=None, hidden_state=[None,None], batch_size=None
                 ):

		decoder = LSTMDecoder(dim=output_dim, hidden_dim=hidden_dim, output_length=output_length,
							  init=init,inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[0],
							  truncate_gradient = truncate_gradient, 
							  hidden_state=hidden_state[0], batch_size=batch_size)

		encoder = LSTMEncoder(input_dim=input_dim, output_dim=hidden_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[2],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[1], batch_size=batch_size, decoder=decoder)
		dense = Dense(input_dim=hidden_dim, output_dim=output_dim)

		if weights[1] is not None:
			dense.set_weights(weights[1])
		self.add(encoder)
		self.add(dense)
		self.add(decoder)

		self.encoder = encoder
		self.dense = dense
		self.decoder = decoder
def get_hidden_state(self):
	return [self.encoder.get_hidden_state(), self.decoder.get_hidden_state()]

def set_hidden_state(self, state):
	self.encoder.set_hidden_state(state[0])
	self.decoder.set_hidden_state(state[1])

def get_weights(self):
	return [l.get_weights() for l in self.layers]

def set_weights(self, weights):
	if len(self.layers) != len(weights):
		raise Exception("Exactly " + str(len(self.layers)) + " weight arrays required " + 
			str(len(weights)) + " given")
	for l,w in zip(l,weights):
		l.set_weights(w)
