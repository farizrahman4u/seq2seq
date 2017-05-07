from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential
from .cells import LSTMDecoderCell, AttentionDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input


'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''


def SimpleSeq2Seq(output_dim, output_length, hidden_dim=None, input_shape=None,
                  batch_size=None, batch_input_shape=None, input_dim=None,
                  input_length=None, depth=1, dropout=0.0, unroll=False,
                  stateful=False):

    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decodes the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence
    elements. The input sequence and output sequence may differ in length.

    Arguments:

    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
            there will be 3 LSTMs on the enoding side and 3 LSTMs on the
            decoding side. You can also specify depth as a tuple. For example,
            if depth = (4, 5), 4 LSTMs will be added to the encoding side and
            5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.

    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size, ) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size, ) + (input_length, ) + (input_dim, )
        else:
            shape = (batch_size, ) + (None, ) + (input_dim)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim
    encoder = RecurrentSequential(unroll=unroll, stateful=stateful)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[-1])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    decoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  decode=True, output_length=output_length)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], hidden_dim)))

    if depth[1] == 1:
        decoder.add(LSTMCell(output_dim))
    else:
        decoder.add(LSTMCell(hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMCell(hidden_dim))


    _input = Input(batch_shape=shape)
    x = encoder(_input)
    output = decoder(x)
    return Model(_input, output)