# -*- coding: UTF-8 -*-
"""
Command Line Interface (CLI) of langdist package.

Usage:
    langdist download-bible <lang-code> <output-corpus-path> [options]
    langdist transliterate <input-corpus-path> <lang-code> <output-corpus-path> [options]
    langdist fit-encoder <encoder-path> <input-corpus-paths>... [options]
    langdist train <input-corpus-path> <encoder-path> <model-path> [options]
    langdist retrain <old-model-path> <input-corpus-path> <model-path> [options]
    langdist generate <model-path> [--sample-num=<int>] [--prompts=<str>] [--top-k=<int>] [--max-len=<int>] [options]
    langdist -h | --help
    langdist -v | --version

Commands:
    download-bible  Download a bible corpus from http://christos-c.com/bible/ and store it into a .pkl file after preprocessing
    transliterate  Transliterate a corpus and store it into a .pkl file
    fit-encoder  Fit an encoder on 1 or more corpora and save it to a .pkl file
    train  Train a language model from the scratch (monolingual model)
    retrain  Train a language model from another language model (bilingual model)
    generate  Generate samples of characters using a trained model

Arguments:
    input-corpus-path  path to the corpus file you want to process
    output-corpus-path  path to where you save(d) the generated corpus 
    encoder-path  path to where you save the fitted encoder
    lang-code  language code (2 characters) of the corpus you want to transliterate (e.g. ar, ja, zh)
    model-path  path to the model directory you where your model will be saved
    old-model-path  path to the model directory of a language model which you want to train a new language model from (only required for `retrain` command)
    
Options:
    # universal options
    -h --help  Show this screen
    -v --version  Show version
    --log-path=<str>  If specified, log into the file at the path
    --verbose  Show debug messages
    
    # options for train command
    --embed-size=<int>  The number of dimensions of the character embedding layer [default: 128] 
    --rnn-size=<int>  The number of dimensions of the RNN layers [default: 256]
    --num-layers=<int>  The number of RNN layers [default: 2]
    --learning-rate=<float>  Initial learning rate of SGD (Adam Optimizer) [default: 0.001]
    --rnn-dropouts=<floats>  Keep probability of dropout in each RNN layer [default: 1.0,1.0]
    --final-dropout=<float>  Keep probability of dropout in the final fully connected layer [default: 1.0]
    
    # options for train/retrain commands
    --batch-size=<int>  The number of samples per batch [default: 128] 
    --patience=<int>  The number of iterations to keep training [default: 819200]
    --valid-size=<float>  The proportion of dataset to use for validation [default: 0.1] 
    --profile  Profile the training (profile_train/valid.json will be created)
    
    # options for generate commands
    --sample-num=<int>  The number of texts to generate [default: 10]
    --prompts=<str>  The first characters which you generate texts from (if None start from empty texts)
    --top-k=<int>  Always sample from top k most probable characters. Set 0 to disable this behaviour. [default: 10]
    --max-len=<int>  The maximum length of characters to generate per text  [default: 300]

Examples:
    langdist download-bible en en_corpus.pkl
    langdist transliterate ja_corpus.pkl ja transliterated_ja_corpus.pkl
    langdist fit-encoder encoder.pkl en_corpus.pkl ja_corpus.pkl zh_corpus.pkl ar_corpus.pkl
    langdist train en_corpus.pkl encoder.pkl en_model --patience=819200 --logpath=langdist.log
    langdist retrain en_model encoder.pkl fr_corpus.pkl en2fr_model --patience=819200 --logpath=langdist.log
    langdist generate en2fr_model --sample-num=50

"""
import json
import os
import shutil
import logging
import pickle
from urllib.request import urlretrieve

from docopt import docopt

from langdist import __version__
from langdist.constant import LANG_CODE2LANGUAGE
from langdist.util import get_logger, set_default_log_path, set_default_log_level, set_log_level, \
    set_log_path
from langdist.preprocess import preprocess_corpus

_BIBLE_CORPUS_URL = 'https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/{}.xml'
_HOME_DIR = '~/'

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er1017@gmail.com'


def download_bible(lang_code, output_corpus_path, keep_xml=False):
    """
    Download a bible corpus from Multilingual Bible Parallel Corpus (http://christos-c.com/bible/),
    perform preprocessing, and save it into a pickle file.
    """
    xml_path = os.path.join(os.path.dirname(output_corpus_path), '{}.xml'.format(lang_code))
    try:
        language = LANG_CODE2LANGUAGE[lang_code]
    except KeyError:
        _LOGGER.error('lang_code={} is not supported. The following lang_codes are supported for '
                      'the corresponding languages:\n{}'
                      .format(lang_code, json.dumps(LANG_CODE2LANGUAGE, indent=2)))
        raise
    urlretrieve(_BIBLE_CORPUS_URL.format(language), xml_path)
    preprocess_corpus(xml_path, output_corpus_path)
    if not keep_xml:
        os.remove(xml_path)


def transliterate(input_corpus_path, lang_code, transliterated_corpus_path):
    """
    Transliterate the text of the given corpus into latin alphabets. `lang_code` needs to be the one
    that is supported by `langdist.transliterator` module.
    """
    from langdist.transliterator import get_transliterator  # import locally because it's slow
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
    from langdist import encoder  # import locally because it's slow to import
    encoder.fit_encoder(input_corpus_paths, encoder_path)


def train(init_args, train_args):
    """Train a language model."""
    from langdist.langmodel import CharLSTM  # import locally because it's slow to import
    char_lstm = CharLSTM(**init_args)
    char_lstm.train(**train_args)


def retrain(old_model_path, train_args):
    """Train a language model on top of the given language model."""
    from langdist.langmodel import CharLSTM  # import locally because it's slow to import
    char_lstm = CharLSTM.load(old_model_path)
    char_lstm.train(**train_args)


def generate(model_path, sample_num, prompts, top_k, max_len):
    """Generate texts using a trained language model."""
    from langdist.langmodel import CharLSTM  # import locally because it's slow to import
    char_lstm = CharLSTM.load(model_path)
    texts = char_lstm.generate(sample_num=sample_num, prompts=prompts, pick_top_k=top_k,
                               max_char_len=max_len)
    print('\n'.join(texts))


def _get_init_args(args):
    """Construct argument dict for CharLSTM.__init__() from args and return it."""
    with open(args['<encoder-path>'], 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return {'embedding_size': int(args['--embed-size']), 'rnn_size': int(args['--rnn-size']),
            'num_rnn_layers': int(args['--num-layers']),
            'learning_rate': float(args['--learning-rate']),
            'rnn_dropouts': [float(dropout) for dropout in args['--rnn-dropouts'].split(',')],
            'final_dropout': float(args['--final-dropout']), 'encoder': encoder}


def _get_train_args(args):
    """Construct argument dict for CharLSTM.train() from args and return it."""
    with open(args['<input-corpus-path>'], 'rb') as input_corpus_file:
        samples = pickle.load(input_corpus_file)
    return {'samples': samples, 'model_path': args['<model-path>'],
            'batch_size': int(args['--batch-size']), 'patience': int(args['--patience']),
            'valid_size': float(args['--valid-size']), 'profile': args['--profile']}


def _expand_user_path(args):
    """Expand to absolute path when ~/ appears in the path."""
    for arg_key, arg_val in args.items():
        if isinstance(arg_val, str) and arg_val.startswith(_HOME_DIR):
            args[arg_key] = os.path.expanduser(arg_val)
    return args


def main():
    """Command line interface for performing various trainings."""
    args = docopt(__doc__, version=__version__)
    args = _expand_user_path(args)

    if args['--verbose']:
        set_default_log_level(logging.DEBUG)
        set_log_level(_LOGGER, logging.DEBUG)

    if args['--log-path']:
        log_path = args['--log-path']
        set_default_log_path(log_path)
        set_log_path(_LOGGER, log_path)

    _LOGGER.debug('Configuration:\n{}'.format(args))

    if args['download-bible']:
        download_bible(args['<lang-code>'], args['<output-corpus-path>'])
        return

    if args['transliterate']:
        transliterate(args['<input-corpus-path>'], args['<lang-code>'],
                      args['<output-corpus-path>'])
        return

    if args['fit-encoder']:
        fit_encoder(args['<input-corpus-paths>'], args['<encoder-path>'])
        return

    if args['generate']:
        generate(args['<model-path>'], int(args['--sample-num']), args['--prompts'],
                 int(args['--top-k']), int(args['--max-len']))
        return

    # set arguments for __init__() and train()
    train_args = _get_train_args(args)

    # remove the model file if already exists
    if os.path.exists(train_args['model_path']):
        shutil.rmtree(train_args['model_path'])

    if args['train']:
        init_args = _get_init_args(args)
        train(init_args, train_args)
    elif args['retrain']:
        retrain(args['<old-model-path>'], train_args)


if __name__ == '__main__':
    main()
