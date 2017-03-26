# -*- coding: UTF-8 -*-
"""
Utility module.
"""
import logging
import os
from logging import getLogger
from xml.dom import minidom

from langdist import PACKAGE_ROOT
from langdist.constant import CORPUS_DIR

__author__ = 'kensk8er1017@gmail.com'

_LOG_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'logs')


class CorpusParser(object):
    """Parser for the parallel multilingual bible corpora (http://christos-c.com/bible/)."""

    def __init__(self, locale):
        self._locale = locale
        self._corpus_path = os.path.join(CORPUS_DIR, '{}.xml'.format(locale))

    def gen_paragraphs(self):
        """Yield paragraph of the corpus."""
        xmldoc = minidom.parse(self._corpus_path)
        segments = xmldoc.getElementsByTagName('seg')
        for segment in segments:
            yield segment.childNodes[0].nodeValue


def get_logger(name, filename=None):
    """Prepare logger for a given name space."""
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file handler
    if filename:
        if not os.path.exists(_LOG_DIR):
            os.mkdir(_LOG_DIR)
        file_handler = logging.FileHandler(os.path.join(_LOG_DIR, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
