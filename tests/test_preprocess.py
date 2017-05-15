# -*- coding: UTF-8 -*-
"""
Unit tests for for preprocess module.
"""
import os
import unittest

import pickle

from langdist.preprocess import preprocess_corpus

_TEST_ROOT = os.path.dirname(__file__)

__author__ = 'kensk8er1017@gmail.com'


class PreprocessTest(unittest.TestCase):
    def test_preprocess_english(self):
        xml_corpus_path = os.path.join(_TEST_ROOT, 'corpora/en.xml')
        processed_corpus_path = os.path.join(_TEST_ROOT, 'en.pkl')

        try:
            preprocess_corpus(xml_corpus_path, processed_corpus_path)

            with open(processed_corpus_path, 'rb') as processed_corpus_file:
                corpus = pickle.load(processed_corpus_file)

            self.assertEqual(len(corpus), 35036)
            self.assertEqual(corpus[0], 'In the beginning God created the heaven and the earth.')
        finally:
            if os.path.exists(processed_corpus_path):
                os.remove(processed_corpus_path)

    def test_preprocess_japanese(self):
        xml_corpus_path = os.path.join(_TEST_ROOT, 'corpora/ja.xml')
        processed_corpus_path = os.path.join(_TEST_ROOT, 'ja.pkl')

        try:
            preprocess_corpus(xml_corpus_path, processed_corpus_path)

            with open(processed_corpus_path, 'rb') as processed_corpus_file:
                corpus = pickle.load(processed_corpus_file)

            self.assertEqual(len(corpus), 31087)
            self.assertEqual(corpus[0], 'はじめに神は天と地とを創造された。')
        finally:
            if os.path.exists(processed_corpus_path):
                os.remove(processed_corpus_path)


if __name__ == '__main__':
    unittest.main()
