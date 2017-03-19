# -*- coding: UTF-8 -*-
"""
This module implements language modeling algorithms.
"""
from copy import deepcopy, copy
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
    _random_state = 0  # this is to make train/test split always return the same split
    _checkpoint_file_name = 'model.ckpt'
    _instance_file_name = 'instance.pkl'

    def __init__(self, embedding_size=128, rnn_size=256, num_rnn_layers=2, learning_rate=0.002,
                 rnn_dropout=0.5, input_dropout=0.8):
        self._embedding_size = embedding_size
        self._rnn_size = rnn_size
        self._num_rnn_layers = num_rnn_layers
        self._learning_rate = learning_rate
        self._rnn_dropout = rnn_dropout
        self._input_dropout = input_dropout
        self._nodes = None
        self._graph = None
        self._vocab_size = None
        self._encoder = CharEncoder()
        self._num_params = None

    def train(self, paragraphs, model_path, batch_size=64, patience=30000, max_iteration=1000000,
              stat_interval=50, valid_interval=300, summary_interval=100, valid_size=0.1):
        """Train a language model on the paragraphs of word IDs."""

        def add_metric_summary(summary_writer, mode, batch_id, perplexity):
            """Add summary for metric."""
            metric_summary = tf.Summary()
            metric_summary.value.add(tag='{}_perplexity'.format(mode), simple_value=perplexity)
            summary_writer.add_summary(metric_summary, global_step=batch_id)

        X = self._encode_chars(paragraphs, fit=True)
        X_train, X_valid = train_test_split(
            X, random_state=self._random_state, test_size=valid_size)

        X_valid, seq_lens_valid = self._add_padding(X_valid)
        X_valid, Y_valid = self._create_Y(X_valid)

        self._build_graph()
        nodes = self._nodes
        train_size = len(X_train)
        train_batch_generator = BatchGenerator(X_train, batch_size)
        best_perplexity = np.float64('inf')

        # Launch the graph
        with tf.Session(graph=self._graph) as session:
            summary_writer = tf.summary.FileWriter(
                os.path.join(model_path, 'tensorboard.log'), session.graph)
            _LOGGER.info('Start fitting a model...')
            session.run(nodes['init'])
            losses = list()

            for batch_id, X_batch in enumerate(train_batch_generator):
                iteration = batch_id * batch_size
                epoch = 1 + iteration // train_size

                if iteration > max_iteration:
                    _LOGGER.info('Iteration is more than max_iteration, finish training.')
                    break

                X_batch, seq_lens = self._add_padding(X_batch)
                X_batch, Y_batch = self._create_Y(X_batch)

                # Predict labels and update the parameters
                _, loss = session.run(
                    [nodes['optimizer'], nodes['loss']],
                    feed_dict={nodes['X']: X_batch, nodes['Y']: Y_batch,
                               nodes['seq_lens']: seq_lens, nodes['is_train']: True})
                losses.append(loss)

                if batch_id > 0 and batch_id % stat_interval == 0:
                    perplexity = np.exp(np.mean(losses))  # cross entropy is log-perplexity
                    _LOGGER.info('Epoch={}, Iter={:,}, Mean Perplexity (Training batch)= {:.3f}'
                                 .format(epoch, iteration, perplexity))
                    losses = list()
                    add_metric_summary(summary_writer, 'train', batch_id, perplexity)

                if batch_id > 0 and batch_id % valid_interval == 0:
                    valid_loss = session.run(
                        nodes['loss'],
                        feed_dict={nodes['X']: X_valid, nodes['Y']: Y_valid,
                                   nodes['seq_lens']: seq_lens_valid, nodes['is_train']: False})
                    perplexity = np.exp(np.mean(valid_loss))
                    _LOGGER.info('Epoch={}, Iter={:,}, Mean Perplexity (Validation set)= {:.3f}'
                                 .format(epoch, iteration, perplexity))
                    add_metric_summary(summary_writer, 'valid', batch_id, perplexity)

                    if perplexity < best_perplexity:
                        _LOGGER.info('Best perplexity so far, save the model.')
                        self._save(model_path, session)
                        best_perplexity = perplexity

                if batch_id > 0 and batch_id % summary_interval == 0:
                    summaries = session.run(nodes['summaries'])
                    summary_writer.add_summary(summaries, global_step=batch_id)

                if iteration >= patience:
                    break

        _LOGGER.info('Finished fitting the model.')
        _LOGGER.info('Best perplexity: {:.3f}'.format(best_perplexity))

    def _save(self, model_path, session):
        """Save the tensorflow session and the instance object of this Python class."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save the session
        self._nodes['saver'].save(session, os.path.join(model_path, self._checkpoint_file_name))

        # save the instance
        instance = copy(self)
        instance._graph = None  # _graph is not picklable
        instance._nodes = None  # _nodes is not pciklable
        with open(os.path.join(model_path, self._instance_file_name), 'wb') as pickle_file:
            pickle.dump(instance, pickle_file)

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
            with tf.name_scope('inputs'):
                nodes['X'] = tf.placeholder(tf.int32, [None, None], name='X')
                nodes['Y'] = tf.placeholder(tf.int32, [None, None], name='Y')
                nodes['seq_lens'] = tf.placeholder(tf.int32, [None], name='seq_lens')
                nodes['is_train'] = tf.placeholder(tf.bool, shape=[], name='is_train')
                input_dropout = tf.where(
                    nodes['is_train'], tf.constant(self._input_dropout), tf.constant(1.0))
                rnn_dropout = tf.where(
                    nodes['is_train'], tf.constant(self._rnn_dropout), tf.constant(1.0))

            # get the shape of the input
            X_shape = tf.shape(nodes['X'])
            batch_size = X_shape[0]
            max_seq_len = X_shape[1]

            with tf.name_scope('embedding_layer'):
                nodes['embeddings'] = tf.Variable(
                    tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                    trainable=True, name='embeddings')
                embedded = tf.nn.embedding_lookup(nodes['embeddings'], nodes['X'])
                embedded = tf.nn.dropout(embedded, input_dropout, name='input_dropout')

            with tf.name_scope('rnn_layer'):
                rnn_cell = LSTMCell(num_units=self._rnn_size)
                nodes['initial_state'] = tf.Variable(
                    rnn_cell.zero_state(batch_size=1, dtype=tf.float32))
                initial_states = tf.split(tf.tile(nodes['initial_state'], [1, batch_size, 1]), 2, 0)
                initial_states = [tf.squeeze(initial_state, [0])
                                  for initial_state in initial_states]
                rnn_cell = DropoutWrapper(rnn_cell, output_keep_prob=rnn_dropout)
                rnn_cell = MultiRNNCell([rnn_cell] * self._num_rnn_layers)
                rnn_outputs, states = tf.nn.dynamic_rnn(
                    rnn_cell, embedded, nodes['seq_lens'], initial_states, tf.float32)

                # reshape rnn_outputs so we can compute activations for all the time steps at once
                rnn_outputs = tf.reshape(rnn_outputs, [-1, self._rnn_size])

            with tf.variable_scope('softmax_layer'):
                nodes['W_s'] = tf.Variable(
                    tf.random_normal([self._rnn_size, self._vocab_size]), name='weight')
                nodes['b_s'] = tf.Variable(tf.random_normal([self._vocab_size]), name='bias')
                logits = tf.matmul(rnn_outputs, nodes['W_s']) + nodes['b_s']

            with tf.variable_scope('optimizer'):
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

            # generate summaries
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                tf.summary.histogram(variable.name, variable)
            nodes['summaries'] = tf.summary.merge_all()

            # save the model to checkpoint
            nodes['saver'] = tf.train.Saver()

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
        """
        Create Y (correct character sequences) based on X (input character sequences). Also prepend
        the paragraph border character to X (in order to learn the beginning of a paragraph.
        """
        Y = list()
        for x in X:
            y = x
            y.append(self._encoder.paragraph_border_id)
            Y.append(y)
            x.insert(0, self._encoder.paragraph_border_id)
        return X, Y

    def _encode_chars(self, paragraphs, fit):
        """Convert paragraphs of characters into encoded characters (character IDs)."""
        if fit:
            encoded_paragraphs = self._encoder.fit_encode(paragraphs)
            self._vocab_size = self._encoder.vocab_size
        else:
            encoded_paragraphs = self._encoder.encode(paragraphs)
        return encoded_paragraphs
