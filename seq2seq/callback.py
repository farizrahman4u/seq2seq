import keras
from keras.callbacks import Callback

class ResetState(Callback):
    """
    This is supposed to be used with stateful RNNs
    Use it for clearing the hidden state after a given number of batches
    Parameters:
    ===========
    rnns: stateful RNNs/ Stateful RNN containers
    func: a function that returns true when the states should be reset.
    states: initial state to which we reset the model when `func` is True.
    """
    def __init__(self, rnns, func, states=None):
        for rnn in rnns:
            if not hasattr(rnn, 'state'):
                raise Exception("Not stateful RNN")
        self.rnns = rnns
        self.states = None
        self.func = func

    def on_batch_end(self, batch, logs={}):
        if self.func(batch, logs):
            if self.states:
                for i in range(len(self.rnns)):
                    self.rnns[i].set_hidden_state(states[i])
            else:
                for rnn in self.rnns:
                    rnn.reset_hidden_state()
