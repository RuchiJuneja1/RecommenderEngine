#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pandas
from typesafety import typesrequired


@typesrequired(pandas.DataFrame)
def is_movielens(datafile):
    columns = ["user", "item", "rate", "st"]

    if len(datafile.columns) != len(columns):
        return False

    if not all([i in datafile.columns for i in columns]):
        return False

    return True


@typesrequired(pandas.DataFrame, int, int)
def remove_rare_elements(data, min_user, min_item):
    if min_user < 0:
        raise ValueError("error! min_user_activity is less than zero")
    if min_item < 0:
        raise ValueError("error! min_item_popularity is less than zero")

    if not is_movielens(data):
        raise ValueError("error! dataframe is not from movielens")

    item_activity = data.groupby('items').size()
    data = data[numpy.in1d(data.items, item_activity[item_activity >= min_item].index)]

    user_activity = data.groupby('users').size()
    data = data[numpy.in1d(data.users, user_activity[user_activity >= min_user].index)]

    return data
