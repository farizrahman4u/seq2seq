# seq2seq
Sequence to Sequence Learning with Keras

**Papers:**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)


![seq2seq](http://i64.tinypic.com/30136te.png)


**Notes:**

* The LSTM Encoder encodes a sequence to a single a vector.
* The LSTM Decoder, when given a hidden state and a vector, generates a sequence.

* In the `Seq2seq` model, the output vector of the LSTM Encoder is the input for the  LSTM Decoder, and
* The hidden state of the LSTM Encoder is copied to the hidden state of LSTM Decoder.


**Example:**

```python
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from seq2seq import Seq2seq
from keras.preprocessing import sequence

vocab_size = 20000 #number of words
maxlen = 100 #length of input sequence and output sequence
embedding_dim = 200 #word embedding size
hidden_dim = 500 #memory size of seq2seq

embedding = Embedding(vocab_size, embedding_dim, input_length=maxlen)
seq2seq = Seq2seq(input_length=maxlen, input_dim=embedding_dim,hidden_dim=hidden_dim,
                  output_dim=embedding_dim, output_length=maxlen, batch_size=10, depth=5)

model = Sequential()
model.add(embedding)
model.add(seq2seq)
```

**Requirements:**

* [Numpy](http://www.numpy.org/)
* [Theano](https://github.com/Theano/Theano) : Do not pip install
* [Keras](keras.io)


**TODO:**

* Add examples with real datasets
