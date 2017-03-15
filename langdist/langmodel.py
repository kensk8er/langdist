# -*- coding: UTF-8 -*-
"""
This module implements language modeling algorithms.
"""
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq import sequence_loss

from langdist.batch import BatchGenerator
from langdist.encoder import CharEncoder
from langdist.util import get_logger

__author__ = 'kensk8er1017@gmail.com'

_LOGGER = get_logger(__name__, write_file=True)


class CharLSTM(object):
    """Character-based language modeling using LSTM."""
    _padding_id = 0  # TODO: 0 is used for actual character as well, which is a bit confusing...

    def __init__(self, embedding_size=128, rnn_size=256, num_rnn_layers=2, learning_rate=0.002):
        self._embedding_size = embedding_size
        self._rnn_size = rnn_size
        self._num_rnn_layers = num_rnn_layers
        self._learning_rate = learning_rate
        self._nodes = None
        self._graph = None
        self._vocab_size = None
        self._encoder = CharEncoder()
        self._num_params = None

    def train(self, paragraphs, batch_size=64, patience=30000, max_iteration=1000000,
              stat_interval=100):
        """Train a language model on the paragraphs of word IDs."""
        X = self._encode_chars(paragraphs, fit=True)
        self._build_graph()
        nodes = self._nodes
        train_size = len(X)
        batch_generator = BatchGenerator(X, batch_size)

        # Launch the graph
        with tf.Session(graph=self._graph) as session:
            _LOGGER.info('Start fitting a model...')
            session.run(nodes['init'])
            losses = list()

            for batch_id, X_batch in enumerate(batch_generator):
                iteration = batch_id * batch_size

                if iteration > max_iteration:
                    _LOGGER.info('Iteration is more than max_iteration, finish training.')
                    break

                X_batch, seq_lens = self._add_padding(X_batch)
                Y_batch = self._create_Y(X_batch)

                # Predict labels and update the parameters
                _, loss = session.run(
                    [nodes['optimizer'], nodes['loss']],
                    feed_dict={nodes['X']: X_batch, nodes['Y']: Y_batch,
                               nodes['seq_lens']: seq_lens, nodes['dropout_prob']: 0.5})
                losses.append(loss)

                if batch_id > 0 and batch_id % stat_interval == 0:
                    epoch = 1 + iteration // train_size
                    _LOGGER.info('Epoch={}, Iter={:,}, Mean Training Loss= {:.3f}'
                                 .format(epoch, iteration, np.mean(losses)))
                    losses = list()

                if iteration >= patience:
                    _LOGGER.info('Early Stopping. No more significant improvement in validation '
                                 'score expected.')
                    break

        _LOGGER.info('Finished fitting the model.')

    def _build_graph(self):
        """Build computational graph."""

        def get_num_params():
            """Count the number of trainable parameters."""
            num_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                var_num_params = 1
                for dimension in shape:
                    var_num_params *= dimension.value
                num_params += var_num_params
            return num_params

        graph = tf.Graph()
        nodes = dict()

        with graph.as_default():
            with tf.name_scope('inputs') as name_scope:
                nodes['X'] = tf.placeholder(tf.int32, [None, None], name='X')
                nodes['Y'] = tf.placeholder(tf.int32, [None, None], name='Y')
                nodes['seq_lens'] = tf.placeholder(tf.int32, [None], name='seq_lens')
                nodes['dropout_prob'] = tf.placeholder(tf.float32, shape=[], name='dropout_prob')

            with tf.name_scope('embedding_layer') as name_scope:
                nodes['embeddings'] = tf.Variable(
                    tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                    trainable=True, name='embeddings')
                embedded = tf.nn.embedding_lookup(nodes['embeddings'], nodes['X'])

            with tf.name_scope('rnn_layer') as name_scope:
                rnn_cell = LSTMCell(num_units=self._rnn_size)
                rnn_cell = DropoutWrapper(rnn_cell, input_keep_prob=nodes['dropout_prob'])
                rnn_cell = MultiRNNCell([rnn_cell] * self._num_rnn_layers)
                rnn_outputs, states = tf.nn.dynamic_rnn(
                    rnn_cell, embedded, dtype=tf.float32, sequence_length=nodes['seq_lens'])

                # reshape rnn_outputs so we can compute activations for all the time steps at once
                rnn_outputs = tf.reshape(rnn_outputs, [-1, self._rnn_size])

            with tf.variable_scope('softmax_layer'):
                nodes['W_s'] = tf.Variable(
                    tf.random_normal([self._rnn_size, self._vocab_size]), name='weight')
                nodes['b_s'] = tf.Variable(tf.random_normal([self._vocab_size]), name='bias')
                logits = tf.matmul(rnn_outputs, nodes['W_s']) + nodes['b_s']

            with tf.variable_scope('loss'):
                # reshape the logits back to batch_size * seq_lens such that we can compute mean
                # loss after masking padding inputs easily by using sequence_loss
                X_shape = tf.shape(nodes['X'])
                batch_size = X_shape[0]
                max_seq_len = X_shape[1]
                logits = tf.reshape(logits, [batch_size, max_seq_len, -1])
                weights = tf.cast(tf.sequence_mask(nodes['seq_lens'], max_seq_len), tf.float32)
                nodes['loss'] = sequence_loss(logits=logits, targets=nodes['Y'], weights=weights)
                nodes['optimizer'] = tf.train.AdamOptimizer(self._learning_rate).minimize(
                    nodes['loss'])

            # initialize the variables
            nodes['init'] = tf.global_variables_initializer()

            # count the number of parameters
            self._num_params = get_num_params()
            _LOGGER.info('Total number of parameters = {:,}'.format(self._num_params))

        self._graph = graph
        self._nodes = nodes

    def _add_padding(self, X):
        """
        Add paddings to X in order to align the sequence lengths.

        :param X: list of sequences of word IDs
        :return: padded list of sequences of word IDs and list of sequence length before padding
        """
        X = deepcopy(X)
        max_len = max(len(x) for x in X)

        seq_lens = list()
        for x in X:
            seq_lens.append(len(x))
            pad_len = max_len - len(x)
            x.extend([self._padding_id for _ in range(pad_len)])
        return X, seq_lens

    def _create_Y(self, X):
        """Create Y (correct character sequences) based on X (input character sequences)."""
        Y = list()
        for x in X:
            y = x[1:]
            y.append(self._encoder.end_symbol_id)
            Y.append(y)
        return Y

    def _encode_chars(self, paragraphs, fit):
        if fit:
            encoded_paragraphs = self._encoder.fit_encode(paragraphs)
            self._vocab_size = self._encoder.vocab_size
        else:
            encoded_paragraphs = self._encoder.encode(paragraphs)
        return encoded_paragraphs
