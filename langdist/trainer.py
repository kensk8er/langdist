# -*- coding: UTF-8 -*-
"""
Command Line Interface for training language models using langdist package.

Usage:
    trainer.py train <locale> [options]
    trainer.py retrain <old-locale> <new-locale> [options]
    trainer.py -h | --help

Commands:
    train  Train a language model from the scratch (monolingual model)
    retrain  Train a language model from another language model (bilingual model)

Options:
    -h --help  Show this screen.
    --patience=<int>  The number of iterations to keep training
    --profile  Profile the training (profile_train/valid.json will be created)

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


def train(locale, train_args):
    """Train a new language model for the given locale."""
    char_lstm = CharLSTM(encoder=get_polyglot_encoder())
    model_path = os.path.join(MODEL_DIR, locale)
    train_args['model_path'] = model_path
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    char_lstm.train(**train_args)


def retrain(old_locale, new_locale, train_args):
    """Retrain a language model that was trained for one locale on a new locale."""
    old_model_path = os.path.join(MODEL_DIR, old_locale)
    char_lstm = CharLSTM.load(old_model_path)
    new_model_path = os.path.join(MODEL_DIR, '{}_{}'.format(old_locale, new_locale))
    train_args['model_path'] = new_model_path
    if os.path.exists(new_model_path):
        shutil.rmtree(new_model_path)
    char_lstm.train(**train_args)


def main():
    """Command line interface for performing various trainings."""
    args = docopt(__doc__)
    _LOGGER.info('Configuration:\n{}'.format(args))
    sentences = load_corpus(args['<locale>']) if args['<locale>'] \
        else load_corpus(args['<new-locale>'])
    train_args = {'sentences': sentences, 'profile': args['--profile']}
    if args['--patience']:
        train_args['patience'] = int(args['--patience'])

    if args['train']:
        train(args['<locale>'], train_args)
    elif args['retrain']:
        retrain(args['<old-locale>'], args['<new-locale>'], train_args)


if __name__ == '__main__':
    main()
