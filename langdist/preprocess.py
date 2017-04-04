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

_SENTENCE_BORDER_REGEX = regex.compile(r'[\.。．!?！？]')
_MAX_PARAGRAPH_LEN = 500


def _preprocess(paragraph, locale):
    """Preprocess a paragraph."""
    paragraph = paragraph.strip()

    # for some reason, zh text has white spaces between characters, which isn't normal for zh texts
    if locale == 'zh':
        paragraph = regex.sub(r'\s', '', paragraph)

    # split into sentences if a paragraph is too long (in order to avoid extremely long run time)
    if len(paragraph) > _MAX_PARAGRAPH_LEN:
        paragraph = _SENTENCE_BORDER_REGEX.split(paragraph)

    return paragraph


class InvalidParagraphException(Exception):
    pass


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
            if isinstance(paragraph, str):
                corpus.append(paragraph)
            elif isinstance(paragraph, list):
                for sentence in paragraph:
                    if sentence and len(sentence) < _MAX_PARAGRAPH_LEN:
                        corpus.append(sentence)
            else:
                raise InvalidParagraphException('paragraph is not str or list.')

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
