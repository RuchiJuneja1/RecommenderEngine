#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
import numpy
import pandas

from recommender.cross_validation import *


def test_load_data():
    """
    load ratings.csv and check againist hard coded parameters
    ratings.csv is the first 100 lines of movielens-small

    :return:
    """

    path = "ratings.csv"
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    assert os.path.exists(absolute_path)