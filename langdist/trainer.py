# -*- coding: UTF-8 -*-
"""
Write training routine here.
"""
import os
import shutil

import pickle

from langdist.constant import MODEL_DIR
from langdist.langmodel import CharLSTM
from langdist.util import get_logger
from langdist.preprocess import load_corpus

__author__ = 'kensk8er1017@gmail.com'

_LOGGER = get_logger(__name__, write_file=True)


def main():
    # TODO: develop CLI and log configuration of each run
    locale = 'zh'
    _LOGGER.info('Locale={}'.format(locale))
    paragraphs = load_corpus(locale)
    universal_encoder = pickle.load(open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb'))
    char_lstm = CharLSTM(encoder=universal_encoder)

    model_path = os.path.join(MODEL_DIR, locale)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    char_lstm.train(paragraphs, model_path=model_path, patience=1000000)


if __name__ == '__main__':
    main()
