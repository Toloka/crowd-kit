#!/usr/bin/env python
# coding: utf8

from setuptools import setup, find_packages

PREFIX = 'crowdkit'

setup(
    name='crowd-kit',
    package_dir={PREFIX: 'src'},
    packages=[f'{PREFIX}.{package}' for package in find_packages('src')],
    version='0.0.2',
    description='Python libraries for crowdsourcing',
    license='Apache 2.0',
    author='Vladimir Losev',
    author_email='losev@yandex-team.ru',
    python_requires='>=3.7.0',
    install_requires=[
        'attrs',
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn',
        'nltk',
    ],
    include_package_data=True,
)
