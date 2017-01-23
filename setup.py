#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Recommender Engine',
    version='prototype',
    description='A recommender engine in tensorflow',
    long_description=readme,
    author='WillBrennan',
    author_email='WillBrennan@users.noreply.github.com',
    url='https://github.com/WillBrennan/SkinDetector',
    license=license,
    install_requires=required,
    packages=find_packages(exclude=('tests', 'docs')))
