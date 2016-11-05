# Utility functions for managing word embeddings
import os.path
import numpy as np


file_name = 'glove.6B.300d.txt'
file_name = os.path.abspath(os.path.join(__file__, os.pardir)) + '/' + file_name


# Mappings
_word_to_index = {}
_index_to_word = []
_index_to_embedding = []

print "Loading embeddings..."

fstream = open(file_name, 'r')
index = 0
for line in fstream:
	vals = line.split()
	word = vals[0]
	emb = np.asarray(vals[1:], dtype='float32')
	_word_to_index[word] = index
	_index_to_word += [word]
	_index_to_embedding += [emb]
	index += 1

embedding_matrix = np.array(_index_to_embedding)
embedding_dim = embedding_matrix.shape[1]
del _index_to_embedding
normalized_embedding_matrix = embedding_matrix / (np.sqrt(np.sum(np.power(embedding_matrix, 2), axis=1, keepdims=True)))

print "Done."

def is_word_in_dict(word):
	return word in _index_to_word

def get_word_from_index(index):
	return _index_to_word[index]

def get_index_from_word(word):
	return _word_to_index[word]

def get_embedding_from_word(word):
	return embedding_matrix[_word_to_index[word]]

def get_embedding_from_index(index):
	return embedding_matrix[index]


def get_normalized_embedding_from_word(word):
	return normalized_embedding_matrix[_word_to_index[word]]

def get_normalized_embedding_from_index(index):
	return normalized_embedding_matrix[index]

def normalize_embedding(embedding):
	return embedding / np.sqrt(np.sum(np.power(embedding, 2)))

def get_match_vector_from_word(word):
	w_emb = normalized_embedding_matrix[_word_to_index[word]]
	return (np.dot(normalized_embedding_matrix, w_emb))

def get_match_vector_from_embedding(embedding):
	embedding = normalize_embedding(embedding)
	return np.dot(normalized_embedding_matrix, embedding)

def get_closest_words_from_word(word, limit=None):
	closest_words = np.argsort(-get_match_vector_from_word(word))
	if limit is not None:
		closest_words = closest_words[:limit]
	return _index_to_word[closest_words]

def get_closest_word_indices_from_word(word, limit=None):
	closest_words = np.argsort(-get_match_vector_from_word(word))
	if limit is not None:
		closest_words = closest_words[:limit]
	return closest_words

def get_closest_words_from_embedding(embedding, limit=None):
	closest_words = np.argsort(-get_match_vector_from_embedding(embedding))
	if limit is not None:
		closest_words = closest_words[:limit]
	return [_index_to_word[i] for i in closest_words]

def get_word_from_embedding(embedding):
	return get_closest_words_from_embedding(embedding, 1)[0]

def analogy(word1, word2, word3):
	emb1, emb2, emb3 = map(get_normalized_embedding_from_word, [word1, word2, word3])
	word3_close_words = get_closest_words_from_embedding(emb3,10)
	emb4 = emb3 + emb2 - emb1
	word4_close_words = get_closest_words_from_embedding(emb4,2)
	if word4_close_words[0] in [word2, word3]:
		return word4_close_words[1]
	else:
		return word4_close_words[0]

assert analogy('man', 'king', 'woman') == 'queen'
