# -*- coding: UTF-8 -*-
"""
Implement Encoder classes that encode characters into character IDs.
"""
import os
import pickle

from sklearn.preprocessing import LabelEncoder

from langdist.constant import LOCALES, MODEL_DIR
from langdist.preprocess import load_corpus

__author__ = 'kensk8er1017@gmail.com'

_ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')


class CharEncoder(object):
    """Encode characters into character IDs."""

    sentence_border = '\n'

    def __init__(self):
        self._label_encoder = LabelEncoder()
        self.sentence_border_id = None
        self._fit = False

    def fit(self, sentences):
        characters = list(''.join(sentences))
        characters.insert(0, self.sentence_border)
        self._label_encoder.fit(characters)
        self.sentence_border_id = int(self._label_encoder.transform([self.sentence_border])[0])
        self._fit = True

    def encode(self, sentences):
        encoded_sentences = list()
        for sentence in sentences:
            encoded_sentences.append(self._label_encoder.transform(list(sentence)).tolist())
        return encoded_sentences

    def decode(self, sentences):
        decoded_sentences = list()
        for sentence in sentences:
            decoded_sentences.append(''.join(self._label_encoder.inverse_transform(sentence)))
        return decoded_sentences

    def fit_encode(self, sentences):
        self.fit(sentences)
        return self.encode(sentences)

    @property
    def vocab_size(self):
        return len(self._label_encoder.classes_)

    def is_fit(self):
        """Return True if the encoder is already fit, else False."""
        return self._fit


def fit_polyglot_encoder(model_path=_ENCODER_PATH):
    """Fit an encoder to all the locales and save it."""
    sentences = list()
    for locale in LOCALES:
        sentences.extend(load_corpus(locale))
    encoder = CharEncoder()
    encoder.fit(sentences)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    with open(model_path, 'wb') as model_file:
        pickle.dump(encoder, model_file)


def get_polyglot_encoder(model_path=_ENCODER_PATH):
    """Return the encoder that is fit to all the locales."""
    if not os.path.exists(model_path):
        fit_polyglot_encoder(model_path)
    with open(model_path, 'rb') as encoder_file:
        return pickle.load(encoder_file)
