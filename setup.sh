#!/usr/bin/env bash
conda create -n langdist python=3.5 pip ipython -y
source activate langdist
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp35-cp35m-linux_x86_64.whl

