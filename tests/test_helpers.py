#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas
import numpy
import recommender


def test_is_movielens():
    columns = ["user", "item", "rate", "st"]
    X = pandas.DataFrame(numpy.random.randn(8, len(columns)), columns=columns)
    assert recommender.helpers.is_movielens(X)

    columns = ["user", "item", "rate", "st", "crap"]
    X = pandas.DataFrame(numpy.random.randn(8, len(columns)), columns=columns)
    assert not recommender.helpers.is_movielens(X)

    columns = ["user", "itm", "rate", "st"]
    X = pandas.DataFrame(numpy.random.randn(8, len(columns)), columns=columns)
    assert not recommender.helpers.is_movielens(X)

    columns = ["user", "rate", "st"]
    X = pandas.DataFrame(numpy.random.randn(8, len(columns)), columns=columns)
    assert not recommender.helpers.is_movielens(X)


def test_remove_rare_elements():
    path = "data/ratings.dat"
    absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    assert os.path.exists(absolute_path)

    datafile = recommender.data.movielens.load_data(absolute_path, "::")

    min_user = 1
    min_item = 2
    new_datafile = recommender.helpers.remove_rare_elements(datafile, min_user, min_item)

    assert recommender.helpers.is_movielens(new_datafile)
    assert new_datafile.shape[0] == 2
    assert new_datafile.shape[1] == 4

    assert "item" in new_datafile.columns

    data_numpy = new_datafile.as_matrix()

    assert data_numpy[0, 0] == 0
    assert data_numpy[1, 0] == 1

    assert data_numpy[0, 1] == 3104
    assert data_numpy[1, 1] == 3104

    assert data_numpy[0, 2] == 5.0
    assert data_numpy[1, 2] == 4.0