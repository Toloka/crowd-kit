#!/usr/bin/env python
# coding: utf8

from setuptools import setup, find_packages

PREFIX = 'crowdkit'

with open('README.md') as f:
    readme = f.read()

setup(
    name='crowd-kit',
    package_dir={PREFIX: 'src'},
    packages=[f'{PREFIX}.{package}' for package in find_packages('src')],
    version='0.0.9',
    description='Python libraries for crowdsourcing',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    author='Vladimir Losev',
    author_email='losev@yandex-team.ru',
    python_requires='>=3.7.0',
    install_requires=[
        'attrs',
        'numpy',
        'pandas >= 1.1.0',
        'tqdm',
        'scikit-learn',
        'nltk',
        'transformers'
    ],
    include_package_data=True,
    project_urls={
        'Source': 'https://github.com/Toloka/crowd-kit',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Typing :: Typed',
    ],
)
