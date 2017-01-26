#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
