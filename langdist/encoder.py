# -*- coding: UTF-8 -*-
"""
Implement Encoder classes that encode characters into character IDs.
"""
from sklearn.preprocessing import LabelEncoder

__author__ = 'kensk8er1017@gmail.com'


class CharEncoder(object):
    """Encode characters into character IDs."""

    end_symbol = '\n'

    def __init__(self):
        self._label_encoder = LabelEncoder()
        self.end_symbol_id = None

    def fit(self, paragraphs):
        paragraphs[0] += self.end_symbol
        self._label_encoder.fit(list(''.join(paragraphs)))
        self.end_symbol_id = self._label_encoder.transform([self.end_symbol])[0]

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
