#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typesafety import typesrequired

import pandas
import numpy


@typesrequired(str)
def load_data(path):
    if not os.path.exists(path):
        raise ValueError("error! path does not exist")

    if not os.path.basename(path) != "ratings.dat":
        raise ValueError("error! this only works on the ratings.dat file from movielens")

    # todo(will) - finish this function

    seperator = "\t"
    col_names = ["user", "item", "rate", "st"]

    datafile = pandas.read_csv(path, sep=seperator, header=None, names=col_names, engine='python')

    datafile["user"] -= 1
    datafile["item"] -= 1

    datafile["user"] = datafile["user"].astype(numpy.int32)
    datafile["item"] = datafile["item"].astype(numpy.int32)

    datafile["rate"] = datafile["rate"].astype(numpy.float32)

    return datafile