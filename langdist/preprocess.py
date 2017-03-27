# -*- coding: UTF-8 -*-
"""
This module is used to preprocess corpora.
"""
import os
import pickle

import regex

from langdist.util import CorpusParser
from langdist.constant import CORPUS_DIR

__author__ = 'kensk8er1017@gmail.com'


def _preprocess(paragraph, locale):
    """Preprocess a paragraph."""
    paragraph = paragraph.strip()

    # for some reason, zh text has white spaces between characters, which isn't normal for zh texts
    if locale == 'zh':
        paragraph = regex.sub(r'\s', '', paragraph)

    return paragraph


def preprocess_corpus(locale):
    """
    Preprocess the corpus of the specified locale and store it into a pickle file.

    :param locale: locale of the corpus to preprocess
    """
    corpus = list()
    parser = CorpusParser(locale)
    for paragraph in parser.gen_paragraphs():
        paragraph = _preprocess(paragraph, locale)
        if paragraph:
            corpus.append(paragraph)

    processed_filepath = os.path.join(CORPUS_DIR, '{}.pkl'.format(locale))
    with open(processed_filepath, 'wb') as processed_file:
        pickle.dump(corpus, processed_file)


def load_corpus(locale):
    """Load corpus for the locale and return paragraphs (list of paragraphs (str))."""
    processed_corpus_path = os.path.join(CORPUS_DIR, '{}.pkl'.format(locale))

    if not os.path.exists(processed_corpus_path):
        preprocess_corpus(locale)

    with open(processed_corpus_path, 'rb') as corpus_file:
        paragraphs = pickle.load(corpus_file)
    return paragraphs
