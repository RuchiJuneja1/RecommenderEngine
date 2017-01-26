#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from typesafety import typesrequired


def split_dataset(data, labels, test_size=0.2, seed=0):
    if test_size > 1.0 or test_size < 0.0:
        raise ValueError("(train_size + test_size) doesnt sum to one")

    n_rows = len(data)
    if n_rows == 0:
        raise ValueError("error! at least one data point is required")

    if n_rows != len(labels):
        raise ValueError("error! data and labels are not the same size")

    random_generator = numpy.random.RandomState(seed)
    permutation = random_generator.permutation(n_rows)

    data = data[permutation]
    labels = labels[permutation]

    n_split = int((1.0 - test_size) * n_rows)

    train_data = data[:n_split]
    test_data = data[n_split:]

    train_labels = labels[:n_split]
    test_labels = labels[n_split:]

    return train_data, test_data, train_labels, test_labels


def split_dataframe(data, test_size=0.2, seed=0):
    if test_size > 1.0 or test_size < 0.0:
        raise ValueError("(train_size + test_size) doesnt sum to one")

    n_rows = len(data)
    if n_rows == 0:
        raise ValueError("error! at least one data point is required")

    random_generator = numpy.random.RandomState(seed)
    permutation = random_generator.permutation(n_rows)

    data = data.iloc[permutation].reset_index(drop=True)

    n_split = int((1.0 - test_size) * n_rows)

    data_train = data[:n_split]
    data_test = data[n_split:].reset_index(drop=True)

    return data_train, data_test
