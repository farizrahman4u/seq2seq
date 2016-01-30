# -*- coding: utf-8 -*-

from __future__ import absolute_import

from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
from seq2seq.layers.bidirectional import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, TimeDistributedDense, Dropout, Activation
from keras.models import Sequential
import theano.tensor as T

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''

class Seq2seqBase(Sequential):
	'''
	Abstract class for all Seq2seq models.
	'''
	wait_for_shape = False

	def add(self, layer):
		'''
		For automatic shape inference in nested models.
		'''
		self.layers.append(layer)
		n = len(self.layers)
		if self.wait_for_shape or (n == 1 and not hasattr(layer, '_input_shape')):
			self.wait_for_shape = True
		elif n > 1:
			layer.set_previous(self.layers[-2])

	def set_previous(self, layer):
		'''
		For automatic shape inference in nested models.
		'''
		self.layers[0].set_previous(layer)
		if self.wait_for_shape:
			self.wait_for_shape = False
			for i in range(1, len(self.layers)):
				self.layers[i].set_previous(self.layers[i - 1])

	def reset_states(self):
		for l in self.layers:
			if  hasattr(l, 'stateful'):
				if l.stateful:
					l.reset_states()

class SimpleSeq2seq(Seq2seqBase):
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
		self.decoder = LSTM(hidden_dim if depth[1]>1 else output_dim, return_sequences=True, **kwargs)
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
		if depth[1] > 1:
			self.add(TimeDistributedDense(output_dim))

class Seq2seq(Seq2seqBase):
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
	def __init__(self, output_dim, hidden_dim, output_length, depth=1, broadcast_state=True, inner_broadcast_state=True, peek=False, dropout=0.25, **kwargs):
		super(Seq2seq, self).__init__()
		layers= []
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		broadcast = (depth[0] > 1 and inner_broadcast_state) or broadcast_state
		encoder = LSTMEncoder(output_dim=hidden_dim, state_input=broadcast, **kwargs)
		if peek:
			decoder = LSTMDecoder2(hidden_dim=hidden_dim, output_length=output_length, state_input=encoder if broadcast else False, **kwargs)
		else:
			decoder = LSTMDecoder(hidden_dim=hidden_dim, output_length=output_length, state_input=encoder if broadcast else False, **kwargs)
		lstms = []
		for i in range(1, depth[0]):
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state and (i != 1), return_sequences=True, **kwargs)
			layers.append(layer)
			lstms.append(layer)
			layers.append(Dropout(dropout))
		layers.append(encoder)
		layers.append(Dropout(dropout))
		layers.append(Dense(hidden_dim if depth[1] > 1 else output_dim))
		lstms.append(encoder)
		if inner_broadcast_state:
			for i in range(len(lstms) - 1):
				lstms[i].broadcast_state(lstms[i + 1])
		layers.append(decoder)
		if broadcast_state:
			encoder.broadcast_state(decoder)
		lstms = [decoder]
		for i in range(1, depth[1]):
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state and (i != 1), return_sequences=True, **kwargs)
			layers.append(layer)
			lstms.append(layer)
			layers.append(Dropout(dropout))
		if inner_broadcast_state:
			for i in range(len(lstms) - 1):
				lstms[i].broadcast_state(lstms[i + 1])
		if depth[1] > 1:
			layers.append(TimeDistributedDense(output_dim))
		self.encoder = encoder
		self.decoder = decoder
		for l in layers:
			self.add(l)
		if depth[0] > 1:
			self.layers[0].build()

class AttentionSeq2seq(Seq2seqBase):

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
	def __init__(self, output_dim, hidden_dim, output_length, depth=1,bidirectional=True, dropout=0.25, **kwargs):
		if bidirectional and hidden_dim % 2 != 0:
			raise Exception ("hidden_dim for AttentionSeq2seq should be even (Because of bidirectional RNN).")
		super(AttentionSeq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		if bidirectional:
			encoder = Bidirectional(LSTMEncoder(output_dim=hidden_dim / 2, state_input=False, return_sequences=True, **kwargs))
		else:
			encoder = LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs)
		decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length, state_input=False, **kwargs)
		lstms = []
		for i in range(1, depth[0]):
			if bidirectional:
				layer = Bidirectional(LSTMEncoder(output_dim=hidden_dim / 2, state_input=False, return_sequences=True, **kwargs))
			else:
				layer = LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs)
			self.add(layer)
			lstms.append(layer)
			self.add(Dropout(dropout))
		self.add(encoder)
		self.add(Dropout(dropout))
		self.add(TimeDistributedDense(hidden_dim if depth[1] > 1 else output_dim))
		lstms.append(encoder)
		self.add(decoder)
		lstms = [decoder]
		for i in range(1, depth[1]):
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs)
			self.add(layer)
			lstms.append(layer)
			self.add(Dropout(dropout))
		if depth[1] > 1:
			self.add(TimeDistributedDense(output_dim))
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
			#kwargs['output_length'] = length
			#kwargs['output_dim'] = length
		if 'input_shape' in kwargs:
			length = kwargs['input_shape'][-2]
			#kwargs['output_length'] = length
			#kwargs['output_dim'] = length
		elif 'batch_input_shape' in kwargs:
			length = kwargs['batch_input_shape'][-2]
			#kwargs['output_length'] = length
			#kwargs['output_dim'] = length
		if 'hidden_dim' not in kwargs:
			kwargs['hidden_dim'] = length
		super(IndexShuffle, self).__init__(output_dim=length, output_length=length, **kwargs)
		self.add(Activation('softmax'))

class SoftShuffle(IndexShuffle):
	'''
	Suffles the timesteps of 3D input. Can also mixup information across timesteps.

	'''
	def get_output(self, train=False):
		indices = super(SoftShuffle, self).get_output(train)
		X = self.get_input(train)
		Y = T.batched_tensordot(indices, X,axes=[(1), (1)])
		return Y

	@property
	def output_shape(self):
	    return self.input_shape
