# -*- coding: UTF-8 -*-
"""
This module is used to preprocess corpora.
"""
import os
import pickle

import regex

from langdist.transliterator import get_transliterator
from langdist.util import CorpusParser
from langdist.constant import CORPUS_DIR, NON_ALPHABET_LOCALES, TRANSLITERATION_CODE

__author__ = 'kensk8er1017@gmail.com'

_SENTENCE_BORDER_REGEX = regex.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s')
_SENTENCE_BORDER_REGEX_ZH = regex.compile(r'\.|\?|\!|。|．|！|？')
_MAX_SENTENCE_LEN = 500


def _sent_tokenize(paragraph, locale):
    """Tokenize paragraph into sentences using a simple regex rule."""
    if locale in ['zh', 'ja']:
        index = 0
        sentences = list()
        for match in _SENTENCE_BORDER_REGEX_ZH.finditer(paragraph):
            sentences.append(paragraph[index: match.end(0)])
            index = match.end(0)

        if index < len(paragraph):
            sentences.append(paragraph[index:])
        return sentences
    else:
        return _SENTENCE_BORDER_REGEX.split(paragraph)


def _preprocess(paragraph, locale):
    """Preprocess corpus text."""
    # for some reason, zh text has white spaces between characters, which isn't normal for zh texts
    if locale == 'zh':
        paragraph = regex.sub(r'\s', '', paragraph)

    return (sentence.strip() for sentence in _sent_tokenize(paragraph, locale))


def preprocess_corpus(locale):
    """
    Preprocess the corpus of the specified locale and store it into a pickle file.

    :param locale: locale of the corpus to preprocess
    """
    corpus = list()
    parser = CorpusParser(locale)
    for paragraph in parser.gen_paragraphs():
        sentences = _preprocess(paragraph, locale)
        for sentence in sentences:
            if sentence and len(sentence) < _MAX_SENTENCE_LEN:
                corpus.append(sentence)

    processed_filepath = os.path.join(CORPUS_DIR, '{}.pkl'.format(locale))
    with open(processed_filepath, 'wb') as processed_file:
        pickle.dump(corpus, processed_file)

    if locale in NON_ALPHABET_LOCALES:
        transliterator = get_transliterator(locale)
        transliterated_corpus = transliterator.transliterate_corpus(corpus)
        transliterated_filepath = os.path.join(
            CORPUS_DIR, '{}-{}.pkl'.format(locale, TRANSLITERATION_CODE))
        with open(transliterated_filepath, 'wb') as processed_file:
            pickle.dump(transliterated_corpus, processed_file)


def load_corpus(locale):
    """Load corpus for the locale and return sentences (list of sentences (str))."""
    processed_corpus_path = os.path.join(CORPUS_DIR, '{}.pkl'.format(locale))

    if not os.path.exists(processed_corpus_path):
        preprocess_corpus(locale)

    with open(processed_corpus_path, 'rb') as corpus_file:
        sentences = pickle.load(corpus_file)
    return sentences
