# -*- coding: utf-8 -*-
from __future__ import absolute_import
from seq2seq.layers.state_transfer_lstm import StateTransferLSTM

class LSTMEncoder(StateTransferLSTM):

	def __init__(self, decoder=None, decoders=[], **kwargs):
		super(LSTMEncoder, self).__init__(**kwargs)
		if decoder:
			decoders = [decoder]
		self.broadcast_state(decoders)
