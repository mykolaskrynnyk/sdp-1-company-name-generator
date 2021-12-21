"""
This script defines two models: a character-level RNN model to be trained on company names
and a CompanyGenerator to be used to produce novel sequences.
"""
# data wrangling
import numpy as np

# deep learning
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense

class CompanyModel(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units: int, rnn_type: str = 'gru'):
        super().__init__(self)

        self.embedding = Embedding(
            input_dim = vocab_size,
            output_dim = embedding_dim,
            mask_zero = True
        )

        # customising an RNN type
        if rnn_type.lower() == 'gru':
            self.rnn = GRU(
                units = rnn_units,
                return_sequences = True,
                return_state = True
            )

        elif rnn_type.lower() == 'lstm':
            self.rnn = LSTM(
                units = rnn_units,
                return_sequences = True,
                return_state = True
            )
        else:
            raise ValueError(f'rnn_type expected either "gru" or "lstm", obtained "{rnn_type}"')

        self.dense = Dense(units = vocab_size)

    def call(self, inputs, states: list = None, return_state: bool = False, training: bool = False):
        x = inputs
        x = self.embedding(x, training = training)

        if states is None:
            states = self.rnn.get_initial_state(x)

        # packing states into an iterable enables to handle either GRU or LSTM
        x, *states = self.rnn(x, initial_state = states, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x

class CompanyGenerator(tf.keras.Model):
    def __init__(self, model, ids2chars, chars2ids):
        super().__init__()

        self.model = model
        self.ids2chars = ids2chars
        self.chars2ids = chars2ids

        # creating a mask not to generate [UNK] character
        mask = np.zeros(len(chars2ids.get_vocabulary()))
        unk_id = chars2ids(['[UNK]']).numpy().item()
        mask[unk_id] = -np.inf
        self.prediction_mask = tf.convert_to_tensor(mask, dtype = tf.float32)

    @tf.function
    def generate_one_step(self, inputs, states = None, temperature: float = 1.0):

        # converting the input into a tensor of ids
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.chars2ids(input_chars).to_tensor()

        # predicting the logits of the next character for each character in the inputs, i.e. [batch, char, next_char_logits]
        y_logit, states = self.model(inputs = input_ids, states = states, return_state = True)

        # subsetting the logits for the next character after the last one
        y_logit = y_logit[:, -1, :]
        y_logit = y_logit / temperature

        # masking the [UNK] token
        y_logit = y_logit + self.prediction_mask

        # sampling for the next character
        y_ids = tf.random.categorical(y_logit, num_samples = 1)
        y_ids = tf.squeeze(y_ids, axis = -1)
        y_hat = self.ids2chars(y_ids)

        return y_hat, states
