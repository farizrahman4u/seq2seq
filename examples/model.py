from seq2seq import *
from keras.models import *
from keras.layers import *
from seq2seq import *


def get_model(**kwargs):
	model = Seq2Seq(output_dim=300, hidden_dim=500, input_dim=300, output_length=5, **kwargs)
	model.compile(loss='mse', optimizer='sgd')
	return model
