from seq2seq import *
import numpy as np



input_length = 10
input_dim = 5

output_length = 8
output_dim = 7


x = np.random.random((100, input_length, input_dim))
y = np.random.random((100, output_length, output_dim))


models = []
models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True)]
models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2)]
models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]

for model in models:
	model.compile(loss='mse', optimizer='sgd')
	model.fit(x, y, nb_epoch=1)
