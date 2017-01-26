import pytest


def test_imports():
    import recommender
    from recommender.cross_validation import split_dataset, split_dataframe
    from recommender import data
    from recommender.data.movielens import load_data
