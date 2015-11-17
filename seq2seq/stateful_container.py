import keras
from keras.models import Sequential
class StatefulContainer(Sequential):
	def __init__(self):
		self.state = []#Mark this container as a stateful layer. Required for nested models. 
		super(StatefulContainer, self).__init__()
	def get_hidden_state(self):
		return [l.get_hidden_state() if hasattr(l, 'state') else None for l in layers]	

	def set_hidden_state(self, state):
		if len(self.layers) != len(state):
			raise Exception("State list length must be equal to number of layers in container")
		for l, s in zip(self.layers, state):
			if hasattr(l, 'state'):
				l.set_hidden_state(s)

	def reset_hidden_state(self):
		for i in range(len(self.layers)):
			if hasattr(self.layers[i], 'state'):
				self.layers[i].set_hidden_state(self.layers[i].get_hidden_state() * 0)

	def get_weights(self):
		return [l.get_weights() for l in self.layers]

	def set_weights(self, weights):
		if len(self.layers) != len(weights):
			raise Exception("Exactly " + str(len(self.layers)) + " weight arrays required " + 
				str(len(weights)) + " given")
		for l,w in zip(l,weights):
			l.set_weights(w)
