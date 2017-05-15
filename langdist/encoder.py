# -*- coding: UTF-8 -*-
"""
Implement Encoder classes that encode characters into character IDs.
"""
import pickle

from sklearn.preprocessing import LabelEncoder

__author__ = 'kensk8er1017@gmail.com'


class CharEncoder(object):
    """Encode characters into character IDs."""

    _segment_char = '\n'  # the character that represents a border between samples

    def __init__(self):
        self._label_encoder = LabelEncoder()
        self._segment_char_id = None
        self._fit = False

    def fit(self, samples):
        """
        Fit the character encoder to the samples of characters given.

        :param samples: samples of characters (e.g. sentences)
        """
        characters = list(''.join(samples))
        characters.insert(0, self._segment_char)
        self._label_encoder.fit(characters)
        self._segment_char_id = int(self._label_encoder.transform([self._segment_char])[0])
        self._fit = True

    def encode(self, samples):
        """
        Encode samples of characters into samples of character IDs using the character encoder.

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of character IDs
        """
        encoded_samples = list()
        for sample in samples:
            encoded_samples.append(self._label_encoder.transform(list(sample)).tolist())
        return encoded_samples

    def decode(self, samples):
        """
        Decode samples of character IDs into samples of original characters using the character 
        encoder. (Reverse operation of encode())

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of original characters
        """
        decoded_samples = list()
        for sample in samples:
            decoded_samples.append(''.join(self._label_encoder.inverse_transform(sample)))
        return decoded_samples

    def fit_encode(self, samples):
        """
        Fit the character encoder to samples of characters and encode them into samples of 
        character IDs using the fitted encoder.

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of character IDs
        """
        self.fit(samples)
        return self.encode(samples)

    @property
    def segment_char(self):
        """The character that represents a border between samples."""
        return self._segment_char

    @property
    def segment_char_id(self):
        """ID of the character that represents a border between samples."""
        return self._segment_char_id

    @property
    def vocab_size(self):
        """The number of unique characters fitted on the encoder."""
        return len(self._label_encoder.classes_)

    @property
    def is_fit(self):
        """True if the encoder is already fit, else False."""
        return self._fit


def fit_encoder(corpus_paths, encoder_path):
    """Fit an encoder to the corpora and save it."""
    corpora = list()
    for corpus_path in corpus_paths:
        with open(corpus_path, 'rb') as corpus_file:
            corpora.extend(pickle.load(corpus_file))

    encoder = CharEncoder()
    encoder.fit(corpora)

    with open(encoder_path, 'wb') as encoder_file:
        pickle.dump(encoder, encoder_file)
