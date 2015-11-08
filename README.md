# seq2seq
Sequence to Sequence Learning with Keras

**Papers:**

* [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)
* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078.pdf)


![seq2seq](http://i64.tinypic.com/30136te.png)


**Notes:**

* The LSTM Encoder encodes a sequence to a single a vector.
* The LSTM Decoder, when given a hidden state and a vector, generates a sequence.

* In the `Seq2seq` model, the output vector of the LSTM Encoder is the input for the  LSTM Decoder, and
* The hidden state of the LSTM Encoder is copied to the hidden state of LSTM Decoder.

**Continious VS Descrete sequence pairs:**

* When training on continuous sequence pairs, such as long conversations, set the `remember_state` argument to `True`for `Seq2seq` model. This is important if you want context sensitive conversational models, so that you can avoid scenarios like this:(Will only work if there are lot of exchanges in each conversation in your training data)

> **Human**: what is your job ?

> **Machine**: i ’m a lawyer .

> **Human**: what do you do ?

> **Machine**: i ’m a doctor

Source : [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)

* Do not forget to clear the hidden state of `Seq2seq` layer after every conversation(**Not after every exchange**) or a fixed number of batches using `reset_hidden_state()` during training and testing.

* In case of descrete sequence pairs(for e.g, machine translation) let the `remeber_state` argument be `False`.


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
                  output_dim=embedding_dim, output_length=maxlen, batch_size=10, depth=4)

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
