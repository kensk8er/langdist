# -*- coding: UTF-8 -*-
"""
Utility module.
"""
import os
from xml.dom import minidom
from logging import getLogger
import logging

from langdist import PACKAGE_ROOT

__author__ = 'kensk8er1017@gmail.com'

_CORPUS_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'corpora')


class CorpusParser(object):
    """Parser for the parallel multilingual bible corpora (http://christos-c.com/bible/)."""

    def __init__(self, locale):
        self._locale = locale
        self._corpus_path = os.path.join(_CORPUS_DIR, '{}.xml'.format(locale))

    def gen_paragraphs(self):
        """Yield paragraph of the corpus."""
        xmldoc = minidom.parse(self._corpus_path)
        segments = xmldoc.getElementsByTagName('seg')
        for segment in segments:
            yield segment.childNodes[0].nodeValue


def get_logger(name):
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
