from embeddings import *
import numpy as np
import os.path
import random

eos = '$'
eos_embedding = get_normalized_embedding_from_word(eos)
max_len = 5

file_name = 'movie_lines.txt'
file_name = os.path.abspath(os.path.join(__file__, os.pardir)) + '/' + file_name


def get_words(x):
	x = x.replace("  ", " ")
	words = x.split(" ")
	if "" in words:
		words.remove("")
	words = [word.lower() for word in words]
	return words


def emb(w):
	if not is_word_in_dict(w):
		return eos_embedding
	return get_normalized_embedding_from_word(w)

with open(file_name, 'r') as f:
	lines = f.readlines()


def get_batch(batch_size):
	if batch_size == -1:
		batch_size = len(lines) - 1
	random_indices = random.sample(range(len(lines) - 1), batch_size)
	X = []
	Y = []
	for i in random_indices:
		X += [map(emb, get_words(lines[i]))]
		Y += [map(emb, get_words(lines[i + 1]))]
	for i in range(batch_size):
		X[i] = X[i][:max_len]
		Y[i] = Y[i][:max_len]
		X[i] += [eos_embedding] * (max_len - len(X[i]))
		Y[i] += [eos_embedding] * (max_len - len(Y[i]))
	return np.array(X), np.array(Y)
