# Seq2seq
Sequence to Sequence Learning with Keras

* *(Latest version of Keras supported : 1.0.5)*


**Hi!** You have just found Seq2seq. Seq2seq is a sequence to sequence learning add-on for the python deep learning library [Keras](http://www.keras.io). Using Seq2seq, you can build and train sequence-to-sequence neural network models in Keras. Such models are useful for machine translation, chatbots (see [[4]](http://arxiv.org/pdf/1506.05869v1.pdf)), parsers, or whatever that comes to your mind.


  ![seq2seq](http://i64.tinypic.com/30136te.png)

# Getting started

Seq2seq contains modular and reusable layers that you can use to build your own seq2seq models as well as built-in models that work out of the box. Seq2seq models can be compiled as they are or added as layers to a bigger model. Every Seq2seq model has 2 primary layers : the encoder and the  decoder. Generally, the encoder encodes the input  sequence to an internal representation called 'context vector' which is used by the decoder to generate the output sequence. The lengths of input and output sequences can be different, as there is no explicit one on one relation between the input and output sequences. In addition to the encoder and decoder layers, a Seq2seq model may also contain layers such as the left-stack (Stacked LSTMs on the encoder side), the right-stack (Stacked LSTMs on the decoder side), resizers (for shape compatibility between the encoder and the decoder) and dropout layers to avoid overfitting. The source code is heavily documented, so lets go straight to the examples:

**A simple Seq2seq model:**

```python
import seq2seq
from seq2seq.models import SimpleSeq2seq

model = SimpleSeq2seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
model.compile(loss='mse', optimizer='rmsprop')
```
That's it! You have successfully compiled a minimal Seq2seq model! Next, let's build a 6 layer deep Seq2seq model (3 layers for encoding, 3 layers for decoding).

**Deep Seq2seq models:**

```python
import seq2seq
from seq2seq.models import SimpleSeq2seq

model = SimpleSeq2seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=3)
model.compile(loss='mse', optimizer='rmsprop')
```
Notice that we have specified the depth for both encoder and decoder as 3, and your model has a total depth of 3 + 3 = 6. You can also specify different depths for the encoder and the decoder. Example:

```python
import seq2seq
from seq2seq.models import SimpleSeq2seq

model = SimpleSeq2seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=20, depth=(4, 5))
model.compile(loss='mse', optimizer='rmsprop')
```

Notice that the depth is specified as tuple, `(4, 5)`. Which means your encoder will be 4 layers deep whereas your decoder will be 5 layers deep. And your model will have a total depth of 4 + 5 = 9.

**Advanced Seq2seq models:**

Until now, you have been using the `SimpleSeq2seq` model, which is a very minimalistic model. In the actual Seq2seq implementation described in [[1]](http://arxiv.org/abs/1409.3215), the hidden state of the encoder is transferred to decoder. Also, the output of decoder at each timestep becomes the input to the decoder at the next time step. To make things more complicated, the hidden state is propogated throughout the LSTM stack. But you  have no reason to worry, as we have a built-in model that does all that out of the box. Example:

```python
import seq2seq
from seq2seq.models import Seq2seq

model = Seq2seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
```

Note that we had to specify the complete input shape, including the samples dimensions. This is because we need a static hidden state(similar to a stateful RNN) for transferring it across layers. By the way, Seq2seq models also support the `stateful` argument, in case you need it.

You can also experiment with the hidden state propogation turned  off. Simply set the arguments `broadcast_state` and `inner_broadcast_state` to `False`.

**Peeky Seq2seq model**:

Let's not stop there. Let's build a model similar to [cho et al 2014](http://arxiv.org/abs/1406.1078), where the decoder gets a 'peek' at the context vector at every timestep.

![cho et al 2014](http://i64.tinypic.com/302aqhi.png)

To achieve this, simply add the argument `peek=True`:

```python
import seq2seq
from seq2seq.models import Seq2seq

model = Seq2seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4, peek=True)
model.compile(loss='mse', optimizer='rmsprop')
```

**Seq2seq model with attention:**

![Attention Seq2seq](http://i64.tinypic.com/a2rw3d.png)

Let's not stop there either. In all the models described above, there is no allignment between the input sequence elements and the output sequence elements. But for machine translation, learning a soft allignment between the input and output sequences imporves performance.[[3]](http://arxiv.org/pdf/1409.0473v6.pdf). The Seq2seq framework includes a ready made attention model which does the same. Note that in the attention model, there is no hidden state propogation, and a bidirectional LSTM encoder is used by default. Example:

```python
import seq2seq
from seq2seq.models import AttentionSeq2seq

model = AttentionSeq2seq(input_dim=5, input_length=7, hidden_dim=10, output_length=8, output_dim=20, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
```

As you can see, in the attention model you need not specify the samples dimension as there are no static hidden states involved(But you have to if you are building a stateful Seq2seq model).
Note:  You  can set the argument `bidirectional=False` if you wish not to use a bidirectional encoder.

# Final Words

That's all for now. Hope you love this library. For any questions you might have, create an issue and I will get in touch. You can also contribute to this project by reporting bugs, adding new examples, datasets or models.

**Installation:**

```sudo pip install git+ssh://github.com/farizrahman4u/seq2seq.git```


**Requirements:**

* [Numpy](http://www.numpy.org/)
* [Theano](https://github.com/Theano/Theano) or [TensorFlow](https/www.tensorflow.org)
* [Keras](keras.io)


**Working Example:**

* [Training Seq2seq with movie subtitles](https://github.com/nicolas-ivanov/debug_seq2seq)  - Thanks to [Nicolas Ivanov](https://github.com/nicolas-ivanov)

**Papers:**

* [ [1] Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [ [2] Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078.pdf)
* [ [3] Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/pdf/1409.0473v6.pdf)
* [ [4] A Neural Conversational Model](http://arxiv.org/pdf/1506.05869v1.pdf)


