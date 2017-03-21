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

    paragraph_border = '\n'

    def __init__(self):
        self._label_encoder = LabelEncoder()
        self.paragraph_border_id = None
        self._fit = False

    def fit(self, paragraphs):
        characters = list(''.join(paragraphs))
        characters.insert(0, self.paragraph_border)
        self._label_encoder.fit(characters)
        self.paragraph_border_id = int(self._label_encoder.transform([self.paragraph_border])[0])
        self._fit = True

    def encode(self, paragraphs):
        encoded_paragraphs = list()
        for paragraph in paragraphs:
            encoded_paragraphs.append(self._label_encoder.transform(list(paragraph)).tolist())
        return encoded_paragraphs

    def decode(self, paragraphs):
        decoded_paragraphs = list()
        for paragraph in paragraphs:
            decoded_paragraphs.append(''.join(self._label_encoder.inverse_transform(paragraph)))
        return decoded_paragraphs

    def fit_encode(self, paragraphs):
        self.fit(paragraphs)
        return self.encode(paragraphs)

    @property
    def vocab_size(self):
        return len(self._label_encoder.classes_)

    def is_fit(self):
        """Return True if the encoder is already fit, else False."""
        return self._fit


def fit_polyglot_encoder(model_path=_ENCODER_PATH):
    """Fit an encoder to all the locales and save it."""
    paragraphs = list()
    for locale in LOCALES:
        paragraphs.extend(load_corpus(locale))
    encoder = CharEncoder()
    encoder.fit(paragraphs)
    with open(model_path, 'wb') as model_file:
        pickle.dump(encoder, model_file)
