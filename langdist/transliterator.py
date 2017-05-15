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

    def transliterate_corpus(self, corpus: list) -> list:
        """
        Transliterate a non-alphabetic corpus into Latin alphabets.

        :param corpus: samples of characters in original scripts
        :return: transliterated corpus
        """
        return [self.transliterate(sample) for sample in corpus]


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


class ArabicTransliterator(BaseTransliterator):
    """
    Transliterate Arabic corpus into Latin alphabets (Buckwalter's scheme ).
    C.f. http://www.qamus.org/transliteration.htm
    """
    _arabic2alphabet = {
        'ء': "'", 'آ': 'a', 'أ': 'a', 'ؤ': "'e", 'إ': 'e', 'ئ': "'e", 'ا': 'A', 'ب': 'b', 'ة': 'p',
        'ت': 't', 'ث': 'v', 'ج': 'g', 'ح': 'H', 'خ': 'x', 'د': 'd', 'ذ': 'd', 'ر': 'r', 'ز': 'z',
        'س': 's', 'ش': 'sh', 'ص': 'S', 'ض': 'D', 'ط': 'T', 'ظ': 'Z', 'ع': 'E', 'غ': 'G', 'ف': 'f',
        'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ى': 'Y', 'ي': 'y',
        'ً': 'F', 'ٌ': 'N', 'ٍ': 'K', 'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ّ': '', 'ْ': 'o'}

    def __init__(self):
        pass

    def transliterate(self, text: str):
        """
        Transliterate Arabic text into Latin alphabets using Buckwalter's scheme.

        :param text: Arabic text
        :return: transliterated latin alphabets
        """
        for arabic, alphabet in self._arabic2alphabet.items():
            text = text.replace(arabic, alphabet)
        return text


def get_transliterator(lang_code, **kwargs):
    """Return transliterator for given locale."""
    lang_code2transliterator_class = {
        'ja': JapaneseTransliterator,
        'zh': ChineseTransliterator,
        'ar': ArabicTransliterator,
    }
    if lang_code not in lang_code2transliterator_class:
        raise NotImplementedError('Transliterator for lang_code={} is not implemented.')
    else:
        return lang_code2transliterator_class[lang_code](**kwargs)
