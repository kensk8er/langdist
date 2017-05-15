# -*- coding: UTF-8 -*-
"""
This module is used to preprocess corpora.
"""
import pickle

import regex

from langdist.util import CorpusParser

__author__ = 'kensk8er1017@gmail.com'

_SENTENCE_BORDER_REGEX = regex.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s')
_SENTENCE_BORDER_REGEX_ZH = regex.compile(r'\.|\?|\!|。|．|！|？')
_MAX_SENTENCE_LEN = 500


def _sent_tokenize(paragraph, lang_code):
    """Tokenize paragraph into sentences using a simple regex rule."""
    if lang_code in ['zh', 'ja']:
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


def _preprocess(paragraph, lang_code):
    """Preprocess corpus text."""
    # for some reason, zh text has white spaces between characters, which isn't normal for zh texts
    if lang_code == 'zh':
        paragraph = regex.sub(r'\s', '', paragraph)

    return (sentence.strip() for sentence in _sent_tokenize(paragraph, lang_code))


def preprocess_corpus(xml_corpus_path, processed_corpus_path):
    """
    Preprocess the raw xml corpus that was downloaded from Multilingual Bible Parallel Corpus
    (http://christos-c.com/bible/) and save it to a .pkl file.

    :param xml_corpus_path: locale of the corpus to preprocess
    :param processed_corpus_path: path to the .pkl file that you save the preprocessed corpus to
    """
    corpus = list()
    parser = CorpusParser(xml_corpus_path)
    lang_code = parser.lang_code

    for paragraph in parser.gen_paragraphs():
        sentences = _preprocess(paragraph, lang_code)
        for sentence in sentences:
            if sentence and len(sentence) < _MAX_SENTENCE_LEN:
                corpus.append(sentence)

    with open(processed_corpus_path, 'wb') as processed_file:
        pickle.dump(corpus, processed_file)
