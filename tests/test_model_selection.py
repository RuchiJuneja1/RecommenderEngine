#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy

from recommender.cross_validation import *


def test_split_dataset_numpy():
    X = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = numpy.array([1, 2, 3, 4, 5])

    X_train, X_test, y_train, y_test = split_dataset(X, y, seed=0)

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

    assert numpy.array_equal(X_train, numpy.array([[5, 6], [1, 2], [3, 4], [7, 8]]))
    assert numpy.array_equal(y_train, numpy.array([3, 1, 2, 4]))

    assert numpy.array_equal(X_test, numpy.array([[9, 10]]))
    assert numpy.array_equal(y_test, numpy.array([5]))


"""
def test_split_dataset_list():
    X = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"]]
    y = [0, 1, 2, 3, 4, 5, 6, 7]

    #X_train, X_test, y_train, y_test = split_dataset(X, y, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2


"""
