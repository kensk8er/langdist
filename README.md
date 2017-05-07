langdist - Character-level Multilingual Language Modeling Toolkit
=================================================================

`langdist` is a Python project for experimenting *Character-level Multilingual Language Modeling*, which is to see how learning a character-level language model in one language helps learning another character-level language model in a different language. The project is still **under development** and can offer limited functionality.


Features
--------
- Download and preprocess multilingual parallel corpora ([Multilingual Bible Parallel Corpus](http://christos-c.com/bible/))
- Train a *monolingual language model*
  - This is a language model trained in one language
- Train a *bilingual language model*
  - This is a language model that is trained on top of another language model (the parameters are initialized using another language model's parameters)
- Sample sentences using a pre-trained language model


Installation
------------
- This repository can run on Ubuntu 14.04 LTS & Mac OSX 10.x (not tested on other OSs)
- Tested only on Python 3.5

This package depends on various 3rd party packages that are tried to be installed via `pip`, but some of the dependencies might fail to be installed depending on your environment. It is recommended to install the following packages before you install `langdist`:

- `numpy>=1.12.0`
- `tensorflow>=1.0.1`
- `scikit-learn>=0.18.1`
- `scipy>=0.18.1`

Simply clone the git repository and run `pip install requirements.txt` on the project root. Installation of `langdist` via `pip` is not provided yet.


Usage
-----
1. Download corpora

CLI for downloading corpora isn't provided yet, but you can edit and run `download-corpora.sh` to download corpora.

2. Training a language model

CLI is provided for training a language model using a corpus downloaded. `python langdist/trainer.py --help` to see the usage of the CLI.


TODO: Add a link to the blog post *Bilingual Character-level Neural Language Modeling*
