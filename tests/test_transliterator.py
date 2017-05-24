# -*- coding: UTF-8 -*-
"""
Unit tests for transliterator module.
"""
import unittest

from langdist.transliterator import get_transliterator

__author__ = 'kensk8er'


class TransliteratorTest(unittest.TestCase):
    def test_japanese_transliterator(self):
        transliterator = get_transliterator('ja')
        japanese = '私の名前は村木です。'
        transliterated = 'Watashi no Namae ha Muraki desu.'
        self.assertEqual(transliterator.transliterate(japanese), transliterated)
        self.assertListEqual(transliterator.transliterate_corpus([japanese]), [transliterated])

    def test_chinese_transliterator(self):
        transliterator = get_transliterator('zh')
        chinese = '我叫村木。'
        transliterated = 'Wo Jiao Cunmu.'
        self.assertEqual(transliterator.transliterate(chinese), transliterated)
        self.assertListEqual(transliterator.transliterate_corpus([chinese]), [transliterated])

    def test_arabic_transliterator(self):
        transliterator = get_transliterator('ar')
        arabic = 'اسمي فؤاد.'
        transliterated = "Asmy f'eAd."
        self.assertEqual(transliterator.transliterate(arabic), transliterated)
        self.assertListEqual(transliterator.transliterate_corpus([arabic]), [transliterated])


if __name__ == '__main__':
    unittest.main()
