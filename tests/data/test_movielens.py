#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
import numpy
import pandas

import recommender


def test_data_exists():
    path = "ratings.dat"
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    assert os.path.exists(absolute_path)


def test_load_data():
    """
    load ratings.dat and check againist hard coded parameters
    ratings.dat is the first 100 lines of movielens-small

    :return:
    """

    path = "ratings.dat"
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    assert os.path.exists(absolute_path)

    datafile = recommender.data.movielens.load_data(absolute_path, "::")

    assert len(datafile) == 100