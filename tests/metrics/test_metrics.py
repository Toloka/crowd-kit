import pandas as pd

from crowdkit.metrics.data import consistency
from crowdkit.metrics.performers import golden_set_accuracy, accuracy_on_aggregates


def test_consistency(request):
    assert consistency(request.getfixturevalue('toy_answers_df')) == 0.9384615384615385


def test_golden_set_accuracy(request):
    answers = request.getfixturevalue('toy_answers_df')
    golden = request.getfixturevalue('toy_gold_df').set_index('task')['label']
    assert golden_set_accuracy(answers, golden) == 5 / 9
    assert golden_set_accuracy(answers, golden, by_performer=True).equals(pd.Series([0.5, 1.0, 1.0, 0.5, 0.0], index=['w1', 'w2', 'w3', 'w4', 'w5'], name='performer'))


def test_accuracy_on_aggregates(request):
    answers = request.getfixturevalue('toy_answers_df')
    assert accuracy_on_aggregates(answers) == 0.7083333333333334
    assert accuracy_on_aggregates(answers, by_performer=True).equals(pd.Series([0.6, 0.8, 1.0, 0.4, 0.8], index=['w1', 'w2', 'w3', 'w4', 'w5'], name='performer'))
