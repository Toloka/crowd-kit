from crowdkit.aggregation import SegmentationEM, SegmentationMajorityVote, SegmentationRASA
from pandas.testing import assert_series_equal
from .data_image import * # noqa:


def test_simple_segmentation_mv(simple_image_df, simple_image_mv_result):
    output = SegmentationMajorityVote().fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_mv_result)


def test_skills_segmentation_mv(image_with_skills_df, image_with_skills_mv_result):
    output = SegmentationMajorityVote().fit_predict(*image_with_skills_df)
    assert_series_equal(output, image_with_skills_mv_result)


def test_simple_segmentation_rasa(simple_image_df, simple_image_rasa_result):
    output = SegmentationRASA().fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_rasa_result)


def test_simple_segmentation_em(simple_image_df, simple_image_em_result):
    output = SegmentationEM().fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_em_result)
