from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test


input_length = 10
input_dim = 2

output_length = 8
output_dim = 3

samples = 100

@keras_test
def test_SimpleSeq2Seq():
	x = np.random.random((samples, input_length, input_dim))
	y = np.random.random((samples, output_length, output_dim))

	models = []
	models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
	models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]

	for model in models:
		model.compile(loss='mse', optimizer='sgd')
		model.fit(x, y, nb_epoch=1)

@keras_test
def test_Seq2Seq():
	x = np.random.random((samples, input_length, input_dim))
	y = np.random.random((samples, output_length, output_dim))

	models = []
	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True)]
	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2)]

	for model in models:
		model.compile(loss='mse', optimizer='sgd')
		model.fit(x, y, nb_epoch=1)

	model = Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2, teacher_force=True)
	model.compile(loss='mse', optimizer='sgd')
	model.fit([x, y], y, nb_epoch=1)
	

@keras_test
def test_AttentionSeq2Seq():
	x = np.random.random((samples, input_length, input_dim))
	y = np.random.random((samples, output_length, output_dim))

	models = []
	models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
	models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
	models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=3)]

	for model in models:
		model.compile(loss='mse', optimizer='sgd')
		model.fit(x, y, nb_epoch=1)
