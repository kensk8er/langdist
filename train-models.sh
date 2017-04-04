#!/usr/bin/env bash

source activate langdist

# train initial language models
python langdist/trainer.py train en
python langdist/trainer.py train fr
python langdist/trainer.py train de
python langdist/trainer.py train ja
python langdist/trainer.py train zh
python langdist/trainer.py train ar
python langdist/trainer.py train pt
python langdist/trainer.py train id

# train other language models on top of initial models
python langdist/trainer.py retrain en fr
python langdist/trainer.py retrain fr en
python langdist/trainer.py retrain en de
python langdist/trainer.py retrain de en
python langdist/trainer.py retrain en ja
python langdist/trainer.py retrain ja en
python langdist/trainer.py retrain ja zh
python langdist/trainer.py retrain zh ja
python langdist/trainer.py retrain en ar
python langdist/trainer.py retrain ar en
python langdist/trainer.py retrain en id
python langdist/trainer.py retrain id en
python langdist/trainer.py retrain ja id
python langdist/trainer.py retrain id ja
