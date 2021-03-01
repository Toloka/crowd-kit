import pytest

from crowdkit.aggregation import HRRASA, RASA

from .data_hrrasa import *  # noqa: F401, F403


@pytest.mark.parametrize(
    'agg_class, predict_method, dataset, results_dataset',
    [
        (HRRASA, 'fit_predict', 'simple', 'hrrasa'),
        (RASA, 'fit_predict', 'simple', 'rasa'),
    ],
    ids=[
        'HRRASA predict outputs on simple dataset',
        'RASA predict outputs on simple dataset',
    ],
)
def test_fit_predict_text_aggregations_methods(
    request, not_random,
    agg_class, predict_method,
    dataset, results_dataset
):
    answers = request.getfixturevalue(f'{dataset}_text_df')
    result = request.getfixturevalue(f'{dataset}_text_result_{results_dataset}')

    aggregator = agg_class()

    somethings_predict = getattr(aggregator, predict_method)(answers)
    assert somethings_predict.equals(result)
