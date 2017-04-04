# -*- coding: UTF-8 -*-
"""
Module to define transliterator classes, which transliterate original corpus into latin alphabets.
"""
import pykakasi

__author__ = 'kensk8er1017@gmail.com'


class JapaneseTransliterator(object):
    """Transliterate Japanese corpus into Latin alphabets."""
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


def get_transliterator(locale, **kwargs):
    """Return transliterator for given locale."""
    locale2transliterator_class = {
        'ja': JapaneseTransliterator,
    }
    if locale not in locale2transliterator_class:
        raise NotImplementedError('Transliterator for locale={} is not implemented.')
    else:
        return locale2transliterator_class[locale](**kwargs)
