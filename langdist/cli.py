# -*- coding: UTF-8 -*-
"""
Command Line Interface of langdist package.

Usage:
    langdist preprocess <input-corpus-path> <output-corpus-path>
    langdist transliterate <input-corpus-path> <lang-code> <output-corpus-path>
    langdist fit-encoder <encoder-path> <input-corpus-path>...
    langdist train <input-corpus-path> <encoder-path> <model-path> [options]
    langdist retrain <old-model-path> <input-corpus-path> <model-path> [options]
    langdist -h | --help
    langdist -v | --version

Commands:
    preprocess  Preprocess a corpus downloaded from http://christos-c.com/bible/ and store it into a .pkl file
    transliterate  Transliterate a preprocessed corpus and store it into a .pkl file
    fit-encoder  Fit an encoder on 1 or more corpora and save it to a .pkl file
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
    input-corpus-path  path to the corpus file you want to process
    output-corpus-path  path to where you save the generated corpus 
    encoder-path  path to where you save the fitted encoder
    lang-code  language code (2 characters) of the corpus you want to transliterate (e.g. ar, ja, zh)
    model-path  path to the model directory you where your model will be saved
    old-model-path  path to the model directory of a language model which you want to train a new language model from (only required for `retrain` command)

Examples:
    langdist preprocess en_corpus.xml en_corpus.pkl
    langdist transliterate ja_corpus.pkl ja transliterated_ja_corpus.pkl
    langdist fit-encoder encoder.pkl en_corpus.pkl ja_corpus.pkl zh_corpus.pkl ar_corpus.pkl
    langdist train en_corpus.pkl encoder.pkl en_model --patience=819200 --logpath=langdist.log
    langdist retrain en_model encoder.pkl fr_corpus.pkl en2fr_model --patience=819200 --logpath=langdist.log

"""
import os
import shutil

import logging

import pickle
from docopt import docopt

from langdist import __version__, encoder
from langdist.langmodel import CharLSTM
from langdist.transliterator import get_transliterator
from langdist.util import get_logger, set_default_log_path, set_default_log_level, set_log_level, \
    set_log_path
from langdist.preprocess import preprocess_corpus

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er1017@gmail.com'


def preprocess(input_corpus_path, output_corpus_path):
    """
    Preprocess a Multilingual Bible Parallel Corpus downloaded from http://christos-c.com/bible/
    and save it into a pickle file.
    """
    preprocess_corpus(input_corpus_path, output_corpus_path)


def transliterate(input_corpus_path, lang_code, transliterated_corpus_path):
    """
    Transliterate the text of the given corpus into latin alphabets. `lang_code` needs to be the one
    that is supported by `langdist.transliterator` module.
    """
    with open(input_corpus_path, 'rb') as input_corpus_file:
        input_corpus = pickle.load(input_corpus_file)
    transliterator = get_transliterator(lang_code)
    transliterated_corpus = transliterator.transliterate_corpus(input_corpus)
    with open(transliterated_corpus_path, 'wb') as transliterated_corpus_file:
        pickle.dump(transliterated_corpus, transliterated_corpus_file)


def fit_encoder(input_corpus_paths, encoder_path):
    """
    Fit an encoder on the corpora given and save it into a pickle file.
    """
    encoder.fit_encoder(input_corpus_paths, encoder_path)


def train(train_args, encoder_path):
    """Train a language model."""
    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    char_lstm = CharLSTM(encoder=encoder)
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

    _LOGGER.debug('Configuration:\n{}'.format(args))

    if args['preprocess']:
        preprocess(args['<input-corpus-path>'], args['<output-corpus-path>'])
        return

    if args['transliterate']:
        transliterate(args['<input-corpus-path>'], args['<lang-code>'],
                      args['<output-corpus-path>'])
        return

    if args['fit-encoder']:
        fit_encoder(args['<input-corpus-path>'], args['<encoder-path>'])
        return

    # set arguments for training
    with open(args['<input-corpus-path>'], 'rb') as input_corpus_file:
        samples = pickle.load(input_corpus_file)
    train_args = {'samples': samples, 'profile': args['--profile'],
                  'model_path': args['<model-path>']}
    if args['--patience']:
        train_args['patience'] = int(args['--patience'])

    # remove the model file if already exists
    if os.path.exists(train_args['model_path']):
        shutil.rmtree(train_args['model_path'])

    if args['train']:
        train(train_args, args['<encoder-path>'])
    elif args['retrain']:
        retrain(args['<old-model-path>'], train_args)


if __name__ == '__main__':
    main()
