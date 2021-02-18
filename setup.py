# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import proseq2vec

VERSION = proseq2vec.__version__
NAME = 'proseq2vec'
DESCRIPTION = 'A library for protein sequence embedding models'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
long_description_content_type = 'text/markdown'
AUTHOR = 'Sameh K. Mohamed'
URL = 'http://samehkamaleldin.github.io/'

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=long_description_content_type,
    author=AUTHOR,
    url=URL,
    install_requires=['numpy',
                      'scikit-learn',
                      ],
    license=LICENSE,
    packages=find_packages(exclude=('tests', 'docs'))
)
