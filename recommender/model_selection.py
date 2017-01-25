#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from typesafety import typesrequired


@typesrequired(numpy.ndarray, float, float, int)
def train_test_validate_split(data, train_size=0.8, test_size=0.1, seed=0):
    if train_size + test_size != 1.0:
        raise ValueError("(train_size + test_size) doesnt sum to one")

    random_generator = numpy.random.RandomState(seed)