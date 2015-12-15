import keras
from keras.models import Sequential
import numpy as np

class StatefulContainer(Sequential):
	def __init__(self):
		self.state = []#Mark this container as a stateful layer. Required for nested models. 
		super(StatefulContainer, self).__init__()
	def get_hidden_state(self):
		state = []
		for l in self.layers:
			if hasattr(l, 'state'):
				state.append += l.get_hidden_state()
			else:
				state += [np.ndarray(0)]
		return state

	def set_hidden_state(self, state):
		for l in self.layers:
			if hasattr(l, 'state'):
				nb_states = len(l.state)
				l.set_hidden_state(state[:nb_states])
				state = state[nb_states:]
			else:
				state = state[1:]

	def reset_hidden_state(self):
		for i in range(len(self.layers)):
			if hasattr(self.layers[i], 'state'):
				self.layers[i].set_hidden_state(map(lambda x:x*0, self.layers[i].get_hidden_state()))

	def get_weights(self):
		weights = super(StatefulContainer, self).get_weights()
		state = self.get_hidden_state()
		return weights + state

	def set_weights(self, weights):
		for  l in self.layers:
			nb_params = len(l.params)
			l.set_weights(weights[:nb_params])
			weights = weights[nb_params:]
		self.set_hidden_state(weights)


