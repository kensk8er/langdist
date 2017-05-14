# -*- coding: UTF-8 -*-
"""
Command Line Interface of langdist package.

Usage:
    langdist preprocess <xml-corpus-path> <processed-corpus-path>
    langdist train <corpus-path> <model-path> [options]
    langdist retrain <old-model-path> <corpus-path> <model-path> [options]
    langdist -h | --help
    langdist -v | --version

Commands:
    preprocess  Preprocess a corpus downloaded from http://christos-c.com/bible/ and store it into a .pkl file
    train  Train a language model from the scratch (monolingual model)
    retrain  Train a language model from another language model (bilingual model)

Options:
    -h --help  Show this screen
    -v --version  Show version
    --patience=<int>  The number of iterations to keep training [default: 819200]
    --profile  Profile the training (profile_train/valid.json will be created)
    --log-path=<str>  If specified, log into the file at the path [default: ]
    --verbose  Show debug messages

Arguments:
    xml-corpus-path  path to the .xml file (downloaded from http://christos-c.com/bible/) you want to preprocess
    processed-corpus-path  path to where you save the generated preprocessed corpus 
    corpus-path  path to the corpus which you want to train a language model on
    model-path  path to the model directory you where your model will be saved
    old-model-path  path to the model directory of a language model which you want to train a new language model from (only required for `retrain` command)

Examples:
    lanbdist preprocess en_corpus.xml en_corpus.pkl
    langdist train en_corpus.pkl en_model --patience=819200 --logpath=langdist.log
    langdist retrain en_model fr_corpus.pkl en2fr_model --patience=819200 --logpath=langdist.log

"""
import os
import shutil

import logging
from docopt import docopt

from langdist import __version__
from langdist.encoder import get_polyglot_encoder
from langdist.langmodel import CharLSTM
from langdist.util import get_logger, set_default_log_path, set_default_log_level, set_log_level, \
    set_log_path
from langdist.preprocess import load_corpus, preprocess_corpus

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er1017@gmail.com'


def preprocess(xml_corpus_path, processed_corpus_path):
    """
    Preprocess a Multilingual Bible Parallel Corpus downloaded from http://christos-c.com/bible/
    and save it into a pickle file.
    """
    preprocess_corpus(xml_corpus_path, processed_corpus_path)


def train(train_args):
    """Train a language model."""
    char_lstm = CharLSTM(encoder=get_polyglot_encoder())
    char_lstm.train(**train_args)


def retrain(old_model_path, train_args):
    """Train a language model on top of the given language model."""
    char_lstm = CharLSTM.load(old_model_path)
    char_lstm.train(**train_args)


def main():
    """Command line interface for performing various trainings."""
    args = docopt(__doc__, version=__version__)

    if args['--verbose']:
        set_default_log_level(logging.DEBUG)
        set_log_level(_LOGGER, logging.DEBUG)

    if args['--log-path']:
        log_path = args['--log-path']
        set_default_log_path(log_path)
        set_log_path(_LOGGER, log_path)

    _LOGGER.info('Configuration:\n{}'.format(args))

    if args['preprocess']:
        preprocess(args['<xml-corpus-path>'], args['<processed-corpus-path>'])
        return

    # set arguments for training
    sentences = load_corpus(args['<corpus-path>'])
    train_args = {'sentences': sentences, 'profile': args['--profile'],
                  'model_path': args['<model-path>']}
    if args['--patience']:
        train_args['patience'] = int(args['--patience'])

    # remove the model file if already exists
    if os.path.exists(train_args['model_path']):
        shutil.rmtree(train_args['model_path'])

    if args['train']:
        train(train_args)
    elif args['retrain']:
        retrain(args['<old-model-path>'], train_args)


if __name__ == '__main__':
    main()
