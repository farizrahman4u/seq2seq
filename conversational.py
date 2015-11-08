import keras
from keras.models import Sequential
from keras.layers.core import Dense

from seq2seq import Seq2seq
from lstm_decoder import LSTMDecoder2


class Conversational(Seq2seq):
	def __init__(self, output_dim, hidden_dim,output_length, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1,
                 input_dim=None, input_length=None, hidden_state=None, batch_size=None, depth=2, context_sensitive=False,
                 ):

		if not type(depth) == list:
			depth = [depth, depth]
		n_lstms = sum(depth)
		if  depth[1] < 2 and context_sensitive = True:
			print "Warning: Your model will not be able to remember its previous output!"
		if weights is None:
			weights = [None] * (n_lstms + 1)

		if hidden_state is None:
			hidden_state = [None] * n_lstms

		encoder_weight_index = depth[0] - 1
		decoder_weight_index = depth[0] + 1

		encoder_state_index = depth[0] - 1
		decoder_state_index = depth[0]

		decoder = LSTMDecoder2(dim=output_dim, hidden_dim=hidden_dim, output_length=output_length,
							  init=init,inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[decoder_weight_index],
							  truncate_gradient = truncate_gradient, 
							  hidden_state=hidden_state[decoder_state_index], batch_size=batch_size, remember_state=context_sensitive)

		encoder = LSTMEncoder(input_dim=input_dim, output_dim=hidden_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[encoder_weight_index],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[encoder_state_index], batch_size=batch_size, remember_state=context_sensitive)

		left_deep = [LSTMEncoder(input_dim=input_dim, output_dim=input_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[i],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[i], batch_size=batch_size, return_sequences=True, remember_state=context_sensitive)
					for i in range(depth[0]-1)]


		right_deep = [LSTMEncoder(input_dim=output_dim, output_dim=output_dim,init=init,
							  inner_init=inner_init, activation=activation, 
							  inner_activation=inner_activation,weights=weights[decoder_weight_index + 1 + i],
							  truncate_gradient = truncate_gradient, input_length=input_length,
							  hidden_state=hidden_state[decoder_state_index + 1 + i], batch_size=batch_size, return_sequences=True, remember_state=context_sensitive)
					for i in range(depth[1]-1)]

		dense = Dense(input_dim=hidden_dim, output_dim=output_dim)
		encoder.decoder = decoder
		if weights[depth[0]] is not None:
			dense.set_weights(weights[depth[0]])
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
