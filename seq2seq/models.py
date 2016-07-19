# -*- coding: utf-8 -*-

from __future__ import absolute_import

from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
from seq2seq.layers.bidirectional import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.engine.topology import Layer


'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


class SimpleSeq2seq(Sequential):
	'''
	Simple model for sequence to sequence learning.
	The encoder encodes the input sequence to vector (called context vector)
	The decoder decoder the context vector in to a sequence of vectors.
	There is no one on one relation between the input and output sequence elements.
	The input sequence and output sequence may differ in length.

	Arguments:

	output_dim : Required output dimension.
	hidden_dim : The dimension of the internal representations of the model.
	output_length : Length of the required output sequence.
	depth : Used to create a deep Seq2seq model. For example, if depth = 3, 
			there will be 3 LSTMs on the enoding side and 3 LSTMs on the 
			decoding side. You can also specify depth as a tuple. For example,
			if depth = (4, 5), 4 LSTMs will be added to the encoding side and
			5 LSTMs will be added to the decoding side.
	dropout : Dropout probability in between layers.

	'''
	def __init__(self, output_dim, hidden_dim, output_length, depth=1, dropout=0.25, **kwargs):
		super(SimpleSeq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		self.encoder = LSTM(hidden_dim, **kwargs)
		self.decoder = LSTM(hidden_dim, return_sequences=True, **kwargs)
		for i in range(1, depth[0]):
			self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
			self.add(Dropout(dropout))
		self.add(self.encoder)
		self.add(Dropout(dropout))
		self.add(RepeatVector(output_length))
		self.add(self.decoder)
		for i in range(1, depth[1]):
			self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
			self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(output_dim)))

class Seq2seq(Sequential):
	'''
	Seq2seq model based on [1] and [2].
	This model has the ability to transfer the encoder hidden state to the decoder's
	hidden state(specified by the broadcast_state argument). Also, in deep models 
	(depth > 1), the hidden state is propogated throughout the LSTM stack(specified by 
	the inner_broadcast_state argument. You can switch between [1] based model and [2] 
	based model using the peek argument.(peek = True for [2], peek = False for [1]).
	When peek = True, the decoder gets a 'peek' at the context vector at every timestep.

	[1] based model:

		Encoder:
		X = Input sequence
		C = LSTM(X); The context vector

		Decoder:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    [2] based model:

		Encoder:
		X = Input sequence
		C = LSTM(X); The context vector

		Decoder:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector 
        from the encoder.

	Arguments:

	output_dim : Required output dimension.
	hidden_dim : The dimension of the internal representations of the model.
	output_length : Length of the required output sequence.
	depth : Used to create a deep Seq2seq model. For example, if depth = 3, 
			there will be 3 LSTMs on the enoding side and 3 LSTMs on the 
			decoding side. You can also specify depth as a tuple. For example,
			if depth = (4, 5), 4 LSTMs will be added to the encoding side and
			5 LSTMs will be added to the decoding side.
	broadcast_state : Specifies whether the hidden state from encoder should be 
					  transfered to the deocder.
	inner_broadcast_state : Specifies whether hidden states should be propogated 
							throughout the LSTM stack in deep models.
	peek : Specifies if the decoder should be able to peek at the context vector
		   at every timestep.
	dropout : Dropout probability in between layers.


	'''
	def __init__(self, output_dim, hidden_dim, output_length, depth=1, broadcast_state=True, inner_broadcast_state=True, peek=False, dropout=0.1, **kwargs):
		super(Seq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		if 'batch_input_shape' in kwargs:
			shape = kwargs['batch_input_shape']
			del kwargs['batch_input_shape']
		elif 'input_shape' in kwargs:
			shape = (None,) + tuple(kwargs['input_shape'])
			del kwargs['input_shape']
		elif 'input_dim' in kwargs:
			shape = (None, None, kwargs['input_dim'])
			del kwargs['input_dim']
		lstms = []
		layer = LSTMEncoder(batch_input_shape=shape, output_dim=hidden_dim, state_input=False, return_sequences=depth[0] > 1, **kwargs)
		self.add(layer)
		lstms += [layer]
		for i in range(depth[0] - 1):
			self.add(Dropout(dropout))
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state, return_sequences=i < depth[0] - 2, **kwargs)
			self.add(layer)
			lstms += [layer]
		if inner_broadcast_state:
			for i in range(len(lstms) - 1):
				lstms[i].broadcast_state(lstms[i + 1])
		encoder = self.layers[-1]
		self.add(Dropout(dropout))
		decoder_type = LSTMDecoder2 if peek else LSTMDecoder
		decoder = decoder_type(hidden_dim=hidden_dim, output_length=output_length, state_input=broadcast_state, **kwargs)
		self.add(decoder)
		lstms = [decoder]
		for i in range(depth[1] - 1):
			self.add(Dropout(dropout))
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state, return_sequences=True, **kwargs)
			self.add(layer)
			lstms += [layer]
		if inner_broadcast_state:
				for i in range(len(lstms) - 1):
					lstms[i].broadcast_state(lstms[i + 1])
		if broadcast_state:
			encoder.broadcast_state(decoder)
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(output_dim)))
		self.encoder = encoder
		self.decoder = decoder


class AttentionSeq2seq(Sequential):

	'''
	This is an attention Seq2seq model based on [3].
	Here, there is a soft allignment between the input and output sequence elements.
	A bidirection encoder is used by default. There is no hidden state transfer in this
	model.

	The  math:

		Encoder:
		X = Input Sequence of length m.
		H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True, 
		so H is a sequence of vectors of length m.

		Decoder:
        y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
        and v (called the context vector) is a weighted sum over H:

        v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

        The weight alpha[i, j] for each hj is computed as follows:
        energy = a(s(i-1), H(j))        
        alhpa = softmax(energy)
        Where a is a feed forward network.

	'''
	def __init__(self, output_dim, hidden_dim, output_length, depth=1,bidirectional=True, dropout=0.1, **kwargs):
		if bidirectional and hidden_dim % 2 != 0:
			raise Exception ("hidden_dim for AttentionSeq2seq should be even (Because of bidirectional RNN).")
		super(AttentionSeq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		if 'batch_input_shape' in kwargs:
			shape = kwargs['batch_input_shape']
			del kwargs['batch_input_shape']
		elif 'input_shape' in kwargs:
			shape = (None,) + tuple(kwargs['input_shape'])
			del kwargs['input_shape']
		elif 'input_dim' in kwargs:
			if 'input_length' in kwargs:
				input_length = kwargs['input_length']
			else:
				input_length = None
			shape = (None, input_length, kwargs['input_dim'])
			del kwargs['input_dim']
		self.add(Layer(batch_input_shape=shape))
		if bidirectional:
			self.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False, return_sequences=True, **kwargs)))
		else:
			self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		for i in range(0, depth[0] - 1):
			self.add(Dropout(dropout))
			if bidirectional:
				self.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False, return_sequences=True, **kwargs)))
			else:
				self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		encoder = self.layers[-1]
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(hidden_dim if depth[1] > 1 else output_dim)))
		decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length, state_input=False, **kwargs)
		self.add(Dropout(dropout))
		self.add(decoder)
		for i in range(0, depth[1] - 1):
			self.add(Dropout(dropout))
			self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(output_dim)))
		self.encoder = encoder
		self.decoder = decoder


class IndexShuffle(SimpleSeq2seq):
	'''
	This model is used for shuffling(re-ordering) the timesteps in 3D data.
	It outputs the shuffled indices of the timesteps.

	'''
	def __init__(self, **kwargs):
		length = None
		if 'input_length' in kwargs:
			length = kwargs['input_length']
		if 'input_shape' in kwargs:
			length = kwargs['input_shape'][-2]
		elif 'batch_input_shape' in kwargs:
			length = kwargs['batch_input_shape'][-2]
		if 'hidden_dim' not in kwargs:
			kwargs['hidden_dim'] = length
		super(IndexShuffle, self).__init__(output_dim=length, output_length=length, **kwargs)
		self.add(Activation('softmax'))

class SoftShuffle(IndexShuffle):
	'''
	Suffles the timesteps of 3D input. Can also mixup information across timesteps.

	'''
	def call(self, x, mask=None):
		import theano.tensor as T
		indices = super(SoftShuffle, self)(x, mask)
		Y = T.batched_tensordot(indices, x,axes=[(1), (1)])
		return Y

	def get_output_shape_for(self, input_shape):
	    return input_shape
