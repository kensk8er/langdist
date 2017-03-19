# -*- coding: UTF-8 -*-
"""
Write training routine here.
"""
import os
import pickle

from langdist import PACKAGE_ROOT
from langdist.langmodel import CharLSTM
from langdist.util import get_logger

__author__ = 'kensk8er1017@gmail.com'

_LOGGER = get_logger(__name__, write_file=True)

_CORPUS_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'corpora')
_MODEL_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'models')


def main():
    # TODO: develop CLI and log configuration of each run
    locale = 'zh'
    _LOGGER.info('Locale={}'.format(locale))
    with open(os.path.join(_CORPUS_DIR, '{}.pkl'.format(locale)), 'rb') as corpus_file:
        paragraphs = pickle.load(corpus_file)

    char_lstm = CharLSTM()
    model_path = os.path.join(_MODEL_DIR, locale)
    char_lstm.train(paragraphs, model_path=model_path, patience=1000000)

if __name__ == '__main__':
    main()
