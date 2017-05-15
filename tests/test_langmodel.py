# -*- coding: UTF-8 -*-
"""
Unit tests for langmodel module.
"""
import os
import unittest

import pickle
import shutil

from langdist.cli import train, retrain

_TEST_ROOT = os.path.dirname(__file__)

__author__ = 'kensk8er1017@gmail.com'


class LangmodelTest(unittest.TestCase):
    def test_train(self):
        corpus_path = os.path.join(_TEST_ROOT, 'corpora/en.pkl')
        model_path = os.path.join(_TEST_ROOT, 'en')
        encoder_path = os.path.join(_TEST_ROOT, 'encoders/en_fr.pkl')
        try:
            with open(corpus_path, 'rb') as corpus_file:
                samples = pickle.load(corpus_file)
            train_args = {'samples': samples, 'model_path': model_path, 'patience': 255}
            train(train_args, encoder_path)
        finally:
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

    def test_retrain(self):
        corpus_path = os.path.join(_TEST_ROOT, 'corpora/fr.pkl')
        old_model_path = os.path.join(_TEST_ROOT, 'models/en')
        model_path = os.path.join(_TEST_ROOT, 'en_fr')
        try:
            with open(corpus_path, 'rb') as corpus_file:
                samples = pickle.load(corpus_file)
            train_args = {'samples': samples, 'model_path': model_path, 'patience': 255}
            retrain(old_model_path, train_args)
        finally:
            if os.path.exists(model_path):
                shutil.rmtree(model_path)


if __name__ == '__main__':
    unittest.main()
