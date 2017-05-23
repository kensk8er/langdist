# -*- coding: UTF-8 -*-
"""
Unit tests for encoder module.
"""
import os
import unittest

import pickle

from langdist.encoder import fit_encoder

_TEST_ROOT = os.path.dirname(__file__)

__author__ = 'kensk8er'


class EncoderTest(unittest.TestCase):
    def test_fit_encoder(self):
        corpus_paths = [os.path.join(_TEST_ROOT, 'corpora/zh.pkl'),
                        os.path.join(_TEST_ROOT, 'corpora/fr.pkl')]
        encoder_path = os.path.join(_TEST_ROOT, 'encoder.pkl')

        try:
            fit_encoder(corpus_paths, encoder_path)

            with open(encoder_path, 'rb') as encoder_file:
                encoder = pickle.load(encoder_file)

            original = ['我叫村木謙介。']
            encoded = encoder.encode(original)
            self.assertEqual(encoded,  [[1058, 438, 1375, 1362, 2654, 150, 90]])
            self.assertEqual(encoder.decode(encoded), original)
        finally:
            if os.path.exists(encoder_path):
                os.remove(encoder_path)


if __name__ == '__main__':
    unittest.main()
