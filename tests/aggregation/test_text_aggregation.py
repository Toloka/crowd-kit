from crowdkit.aggregation import RASA, HRRASA
from pandas.testing import assert_series_equal
from .data_hrrasa import *  # noqa:


def test_rasa(simple_text_df, simple_text_true_embeddings, simple_text_result_rasa):
    output = RASA().fit_predict(simple_text_df, simple_text_true_embeddings)
    assert_series_equal(output, simple_text_result_rasa)


def test_hrrasa(simple_text_df, simple_text_true_embeddings, simple_text_result_hrrasa):
    output = HRRASA().fit_predict(simple_text_df, simple_text_true_embeddings)
    assert_series_equal(output, simple_text_result_hrrasa)
