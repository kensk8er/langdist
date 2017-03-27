# -*- coding: UTF-8 -*-
"""
Define constants used across langdist package in this module.
"""
import os

from langdist import PACKAGE_ROOT

__author__ = 'kensk8er1017@gmail.com'

LOCALES = ['en', 'zh', 'ja', 'fr', 'de', 'pt', 'ind', 'ar']
CORPUS_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'corpora')
MODEL_DIR = os.path.join(PACKAGE_ROOT, os.path.pardir, 'models')
