# Deep LSTM with hidden state forwarding
import keras
from keras.models import Sequential
from lstm_encoder import LSTMEncoder as lstm
from keras.layers.core import RepeatVector
from stateful_container import StatefulContainer

class DeepLSTM(StatefulContainer):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, input_length=None, hidden_state=None, batch_size=None, return_sequences = False, inner_return_sequences = False, remember_state=False, **kwargs):
    	if depth < 1:
    		raise Exception("Minimum depth is 1")
    	if depth == 1:
    		nlayer = depth
    	elif not inner_return_sequences:
    		nlayer = depth*2 - 1

    	if weights is None:
    		weights = [None]*nlayer
    	if hidden_state is None:
    		hidden_state = [None]*depth

    	super(DeepLSTM, self).__init__()
    	def get_lstm(idim, odim, rs, i):
    		return lstm(input_dim=idim, output_dim=odim, init=init,
    				inner_init=inner_init, forget_bias_init=forget_bias_init,
    				activation=activation, inner_activation=inner_activation,
    				weights=weights[i], truncate_gradient=truncate_gradient,
    				input_length=input_length, hidden_state=hidden_state[i],
    				batch_size=batch_size, return_sequences=rs,
    				remember_state=remember_state, **kwargs)
    	if depth == 1:
    		self.add(get_lstm(input_dim, output_dim, return_sequences, 0))
    	else:
    		lstms = []

    		layer = get_lstm(input_dim, hidden_dim, inner_return_sequences, 0)
    		lstms.append(layer)
    		self.add(layer)
    		if not inner_return_sequences:
    			self.add(RepeatVector(input_length))

    		for i in range(depth-2):
    			layer = get_lstm(hidden_dim, hidden_dim, inner_return_sequences, i+1)
       			lstms.append(layer)
    			self.add(layer)
    			if not inner_return_sequences:
    				self.add(RepeatVector(input_length))

    		layer = get_lstm(hidden_dim, output_dim, return_sequences)
    		lstms.append(layer)
    		self.add(layer)

    		for i in range(len(lstms)-1):#connect hidden layers.
    			lstms[i].broadcast_state(lstms[i+1])


