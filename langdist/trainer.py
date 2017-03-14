# -*- coding: UTF-8 -*-
"""
Write training routine here.
"""
import os
import pickle

from langdist import PACKAGE_ROOT
from langdist.langmodel import CharLSTM

__author__ = 'kensk8er1017@gmail.com'

_CORPUS_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'corpora')


def main():
    # TODO: develop CLI and log configuration of each run
    with open(os.path.join(_CORPUS_DIR, 'en.pkl'), 'rb') as corpus_file:
        paragraphs = pickle.load(corpus_file)

    char_lstm = CharLSTM()
    char_lstm.train(paragraphs, patience=1000000)

if __name__ == '__main__':
    main()
