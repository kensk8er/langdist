# -*- coding: UTF-8 -*-
"""
Command Line Interface for training language models using langdist package.

Usage:
    trainer.py train <locale>
    trainer.py retrain <old-locale> <new-locale>
    trainer.py -h | --help

Options:
    -h --help  Show this screen.

Arguments:
    locale  locale of which you want to train a language model (only required for `train` command)
    old-locale  locale of the language model which you want to train a new language model from (only required for `retrain` command)
    new-locale  locale of which you want to retrain a new language model from an old language model
"""
import os
import shutil

from docopt import docopt

from langdist import langmodel
from langdist.constant import MODEL_DIR
from langdist.encoder import get_polyglot_encoder
from langdist.langmodel import CharLSTM
from langdist.util import get_logger
from langdist.preprocess import load_corpus

__author__ = 'kensk8er1017@gmail.com'

_LOGGER = get_logger(__name__, filename='{}.log'.format(langmodel.__name__))


def train(args):
    """Train a new language model for the given locale."""
    _LOGGER.info('Configuration:\n{}'.format(args))
    paragraphs = load_corpus(args['<locale>'])
    char_lstm = CharLSTM(encoder=get_polyglot_encoder())
    model_path = os.path.join(MODEL_DIR, args['<locale>'])
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    char_lstm.train(paragraphs, model_path=model_path)


def retrain(args):
    """Retrain a language model that was trained for one locale on a new locale."""
    _LOGGER.info('Configuration:\n{}'.format(args))
    paragraphs = load_corpus(args['<new-locale>'])
    old_model_path = os.path.join(MODEL_DIR, args['<old-locale>'])
    char_lstm = CharLSTM.load(old_model_path)
    new_model_path = os.path.join(
        MODEL_DIR, '{}_{}'.format(args['<old-locale>'], args['<new-locale>']))
    if os.path.exists(new_model_path):
        shutil.rmtree(new_model_path)
    char_lstm.train(paragraphs, model_path=new_model_path)


def main():
    """Command line interface for performing various trainings."""
    args = docopt(__doc__)

    if args['train']:
        train(args)
    elif args['retrain']:
        retrain(args)


if __name__ == '__main__':
    main()
