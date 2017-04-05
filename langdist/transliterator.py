# -*- coding: UTF-8 -*-
"""
Module to define transliterator classes, which transliterate original corpus into latin alphabets.
"""
from abc import ABCMeta, abstractmethod

import jieba
import pinyin
import pykakasi

__author__ = 'kensk8er1017@gmail.com'


class BaseTransliterator(metaclass=ABCMeta):
    """Base class of transliterators."""

    @abstractmethod
    def transliterate(self, text: str) -> str:
        pass

    @abstractmethod
    def transliterate_corpus(self, corpus: list) -> list:
        pass


class JapaneseTransliterator(BaseTransliterator):
    """Transliterate Japanese corpus into Latin alphabets (romaji)."""
    _space = ' '
    _invalid_chars = ['々']

    def __init__(self, space=True, capitalize=True, convert_symbol=True):
        """
        :param space: add space between words if set True
        :param capitalize: capitalize words (except grammatical words)
        :param convert_symbol: convert symbols to latin-alphabet equivalents
        """
        kakasi = pykakasi.kakasi()
        kakasi.setMode('H', 'a')
        kakasi.setMode('K', 'a')
        kakasi.setMode('J', 'a')

        if space:
            kakasi.setMode('s', True)
            kakasi.setMode('S', self._space)
        kakasi.setMode('C', capitalize)
        if convert_symbol:
            kakasi.setMode('E', 'a')

        self._converter = kakasi.getConverter()

    def transliterate(self, text: str) -> str:
        """
        Transliterate Japanese text into Latin alphabets.

        :param text: Japanese text
        :return: transliterated latin alphabets
        """
        try:
            return self._converter.do(text)
        except TypeError:
            for invalid_char in self._invalid_chars:
                text = text.replace(invalid_char, '')
            return self._converter.do(text)

    def transliterate_corpus(self, corpus: list) -> list:
        """
        Transliterate Japanese corpus into Latin alphabets.

        :param corpus: list of Japanese sentences
        :return: transliterated corpus
        """
        return [self.transliterate(sentence) for sentence in corpus]


class ChineseTransliterator(BaseTransliterator):
    """Transliterate Chinese corpus into Latin alphabets (pinyin)."""
    _space = ' '
    _symbols = {'。': '.', '、': ',', '！': '!', '？': '?'}

    def __init__(self, space=True, capitalize=True, convert_symbol=True):
        """
        :param space: add space between words if set True
        :param capitalize: capitalize words
        :param convert_symbol: convert symbols to latin-alphabet equivalents if possible
        """
        self._add_space = space
        self._capitalize = capitalize
        self._convert_symbol = convert_symbol

    def transliterate(self, text: str) -> str:
        """
        Transliterate Chinese text into Latin alphabets.

        :param text: Chinese text
        :return: transliterated latin alphabets
        """
        words = (pinyin.get(word, format='strip') for word in jieba.cut(text))

        if self._convert_symbol:
            words = (self._symbols.get(word, word) for word in words)
        if self._capitalize:
            words = (word.capitalize() for word in words)
        join_char = self._space if self._add_space else ''
        text = join_char.join(words)

        if self._space:
            symbols = self._symbols.values() if self._convert_symbol else self._symbols.keys()
            for symbol in symbols:
                text = text.replace(' {}'.format(symbol), symbol)

        return text

    def transliterate_corpus(self, corpus: list) -> list:
        """
        Transliterate Chinese corpus into Latin alphabets.

        :param corpus: list of Chinese sentences
        :return: transliterated corpus
        """
        return [self.transliterate(sentence) for sentence in corpus]


def get_transliterator(locale, **kwargs):
    """Return transliterator for given locale."""
    locale2transliterator_class = {
        'ja': JapaneseTransliterator,
        'zh': ChineseTransliterator,
    }
    if locale not in locale2transliterator_class:
        raise NotImplementedError('Transliterator for locale={} is not implemented.')
    else:
        return locale2transliterator_class[locale](**kwargs)
