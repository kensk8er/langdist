# -*- coding: UTF-8 -*-
"""
Utility module.
"""
import logging
import os
from logging import getLogger
from xml.dom import minidom

_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_DEFAULT_LOG_PATH = None  # don't write to a file in default
_DEFAULT_LOG_LEVEL = logging.INFO

__author__ = 'kensk8er1017@gmail.com'


class CorpusParser(object):
    """Parser for the parallel multilingual bible corpora (http://christos-c.com/bible/)."""
    _language_tag = 'language'
    _id_attribute = 'id'
    _segment_tag = 'seg'

    def __init__(self, corpus_path):
        self._lang_code = None
        self._corpus_path = corpus_path

    @property
    def lang_code(self):
        """Return the language code of the corpus."""
        if not self._lang_code:
            xmldoc = minidom.parse(self._corpus_path)
            self._lang_code = xmldoc.getElementsByTagName(
                self._language_tag)[0].attributes[self._id_attribute].value
        return self._lang_code

    def gen_paragraphs(self):
        """Yield paragraph of the corpus."""
        xmldoc = minidom.parse(self._corpus_path)
        segments = xmldoc.getElementsByTagName(self._segment_tag)
        for segment in segments:
            if segment.childNodes:
                yield segment.childNodes[0].nodeValue


def get_logger(name, filepath=None, log_level=None):
    """Prepare logger for a given name space."""
    log_level = log_level or _DEFAULT_LOG_LEVEL
    logger = getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(_LOG_FORMAT)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file handler
    filepath = filepath or _DEFAULT_LOG_PATH
    if filepath:
        set_log_path(logger, filepath, log_level)

    return logger


def set_log_path(logger, filepath, log_level=None):
    """Set FileHandler with the filepath for the given logger."""
    log_level = log_level or _DEFAULT_LOG_LEVEL
    formatter = logging.Formatter(_LOG_FORMAT)

    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # remove existing FileHandlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def set_log_level(logger, log_level: int):
    """Set log level for the given logger."""
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


def set_default_log_path(filepath: str):
    """Set the default path for log files."""
    global _DEFAULT_LOG_PATH
    _DEFAULT_LOG_PATH = filepath


def set_default_log_level(log_level: int):
    """Set the default log level for logging."""
    global _DEFAULT_LOG_LEVEL
    _DEFAULT_LOG_LEVEL = log_level
