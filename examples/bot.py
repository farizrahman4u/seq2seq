from model import get_model
from data.embeddings import *
from data import get_words, emb
import numpy as np
import keras.backend as K
import os.path

file_name = 'weights.dat'
file_name = os.path.abspath(os.path.join(__file__, os.pardir)) + '/' + file_name


model = get_model(dropout=0.)
model.load_weights(file_name)

# force compile
model.predict(np.zeros((10, 50, 300)))

while True:
	input_text = raw_input('Type your message : ')
	input_text = get_words(input_text)
	input_text = input_text[:50]
	input_text += ['$'] * (50 - len(input_text))
	input_text = map(emb, input_text)
	input_text = np.array(input_text)
	input_text = np.expand_dims(input_text, 0)
	output = model.predict(input_text)[0]
	output = [get_word_from_embedding(output[i]) for i in range(len(output))]
	output_text = ''
	for w in output:
		output_text += w + ' '
	output_text = output_text[:-1]
	print('Bot : ' + output_text)

