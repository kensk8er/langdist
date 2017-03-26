#!/usr/bin/env bash

source activate langdist

# train initial language models
python langdist/trainer.py train en --patience=30000
python langdist/trainer.py train fr --patience=30000
python langdist/trainer.py train de --patience=30000
python langdist/trainer.py train ja --patience=30000
python langdist/trainer.py train zh --patience=30000
python langdist/trainer.py train ar --patience=30000

# train other language models on top of initial models
python langdist/trainer.py retrain en fr --patience=30000
python langdist/trainer.py retrain fr en --patience=30000
python langdist/trainer.py retrain en de --patience=30000
python langdist/trainer.py retrain de en --patience=30000
python langdist/trainer.py retrain en ja --patience=30000
python langdist/trainer.py retrain ja en --patience=30000
python langdist/trainer.py retrain ja zh --patience=30000
python langdist/trainer.py retrain zh ja --patience=30000
python langdist/trainer.py retrain en ar --patience=30000
python langdist/trainer.py retrain ar en --patience=30000
