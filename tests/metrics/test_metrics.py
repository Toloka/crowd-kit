import pandas as pd
import numpy as np
import pytest
from nltk.metrics.distance import masi_distance
from pandas.testing import assert_series_equal

from crowdkit.aggregation.utils import get_accuracy
from crowdkit.metrics.data import alpha_krippendorff, consistency, uncertainty
from crowdkit.metrics.workers import accuracy_on_aggregates


def test_consistency(toy_answers_df: pd.DataFrame) -> None:
    assert consistency(toy_answers_df) == 0.9384615384615385


class TestUncertaintyMetric:
    def test_uncertainty_mean_per_task_skills(self, toy_answers_df: pd.DataFrame) -> None:
        workers_skills = pd.Series(
            [0.6, 0.8, 1.0,  0.4, 0.8],
            index=pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
        )

        assert uncertainty(toy_answers_df, workers_skills) == 0.6308666201949331

    def test_uncertainty_raises_wrong_compte_by(self, toy_answers_df: pd.DataFrame) -> None:
        workers_skills = pd.Series(
            [0.6, 0.8, 1.0,  0.4, 0.8],
            index=pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
        )
        with pytest.raises(KeyError):
            uncertainty(toy_answers_df, workers_skills, compute_by='invalid')

    def test_uncertainty_docstring_examples(self) -> None:
        assert uncertainty(
            pd.DataFrame.from_records(
                [
                    {'task': 'X', 'worker': 'A', 'label': 'Yes'},
                    {'task': 'X', 'worker': 'B', 'label': 'Yes'},
                ]
            )
        ) == 0.0

        assert uncertainty(
            pd.DataFrame.from_records(
                [
                    {'task': 'X', 'worker': 'A', 'label': 'Yes'},
                    {'task': 'X', 'worker': 'B', 'label': 'No'},
                    {'task': 'X', 'worker': 'C', 'label': 'Maybe'},
                ]
            )
        ) == 1.0986122886681096

        np.testing.assert_allclose(  # type: ignore
            uncertainty(
                pd.DataFrame.from_records(
                    [
                        {'task': 'X', 'worker': 'A', 'label': 'Yes'},
                        {'task': 'X', 'worker': 'B', 'label': 'No'},
                        {'task': 'Y', 'worker': 'A', 'label': 'Yes'},
                        {'task': 'Y', 'worker': 'B', 'label': 'Yes'},
                    ]
                ),
                compute_by="task",
                aggregate=False
            ), pd.Series([0.693147, 0.0], index=['X', 'Y'], name='task'), atol=1e-3
        )

        np.testing.assert_allclose(  # type: ignore
            uncertainty(
                pd.DataFrame.from_records(
                    [
                        {'task': 'X', 'worker': 'A', 'label': 'Yes'},
                        {'task': 'X', 'worker': 'B', 'label': 'No'},
                        {'task': 'Y', 'worker': 'A', 'label': 'Yes'},
                        {'task': 'Y', 'worker': 'B', 'label': 'Yes'},
                    ]
                ),
                compute_by="worker",
                aggregate=False
            ), pd.Series([0.0, 0.693147], index=['A', 'B'], name='worker'), atol=1e-3
        )

    def test_uncertainty_raises_skills_not_found(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'C', 'label': frozenset(['cat'])},
            ]
        )

        workers_skills = pd.Series(
            [1, 1],
            index=pd.Index(['A', 'B'], name='worker'),
        )

        with pytest.raises(AssertionError):
            uncertainty(answers, workers_skills)

    def test_uncertainty_per_worker(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'C', 'label': frozenset(['dog'])},
                {'task': '4', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'C', 'label': frozenset(['cat'])},
            ]
        )

        workers_skills = pd.Series(
            [1, 1, 1],
            index=pd.Index(['A', 'B', 'C'], name='worker'),
        )

        entropies = uncertainty(
            answers,
            workers_skills,
            compute_by='worker',
            aggregate=False
        )

        assert isinstance(entropies, pd.Series)
        assert sorted(np.unique(entropies.index).tolist()) == ['A', 'B', 'C']  # type: ignore

        # B always answers the same, entropy = 0
        np.testing.assert_allclose(entropies['B'], 0, atol=1e-6)  # type: ignore

        # A answers uniformly, entropy = max possible
        np.testing.assert_allclose(entropies['A'], 0.693147, atol=1e-6)  # type: ignore

        # C answers non-uniformly, entropy = between B and A
        assert entropies['A'] > entropies['C'] > entropies['B']

        assert entropies.mean() == uncertainty(
            answers,
            workers_skills,
            compute_by='worker',
            aggregate=True
        )

    def test_uncertainty_per_task(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'C', 'label': frozenset(['dog'])},
                {'task': '4', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '5', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '5', 'worker': 'B', 'label': frozenset(['dog'])},
            ]
        )

        workers_skills = pd.Series(
            [1, 1, 1],
            index=pd.Index(['A', 'B', 'C'], name='worker'),
        )

        entropies = uncertainty(answers,
                                workers_skills,
                                compute_by='task',
                                aggregate=False)

        assert isinstance(entropies, pd.Series)
        assert sorted(np.unique(entropies.index).tolist()) == ['1', '2', '3', '4', '5']  # type: ignore

        # Everybody answered same on tasks 2 and 4
        np.testing.assert_allclose(entropies['2'], 0, atol=1e-6)  # type: ignore
        np.testing.assert_allclose(entropies['4'], 0, atol=1e-6)  # type: ignore

        # On tasks 1 and 3, 2 workers agreed and one answered differently
        assert entropies['1'] > 0
        np.testing.assert_allclose(entropies['1'], entropies['3'], atol=1e-6)  # type: ignore

        # Complete disagreement on task 5, max possible entropy
        np.testing.assert_allclose(entropies['5'], 0.693147, atol=1e-6)  # type: ignore

        assert entropies.mean() == uncertainty(
            answers,
            workers_skills,
            compute_by='task',
            aggregate=True
        )


def test_golden_set_accuracy(toy_answers_df: pd.DataFrame, toy_gold_df: pd.Series) -> None:
    assert get_accuracy(toy_answers_df, toy_gold_df) == 5 / 9
    assert get_accuracy(toy_answers_df, toy_gold_df, by='worker').equals(pd.Series(
        [0.5, 1.0, 1.0, 0.5, 0.0],
        index=['w1', 'w2', 'w3', 'w4', 'w5'],
    ))


def test_accuracy_on_aggregates(toy_answers_df: pd.DataFrame) -> None:
    expected_workers_accuracy = pd.Series(
        [0.6, 0.8, 1.0,  0.4, 0.8],
        index=pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
    )
    assert_series_equal(accuracy_on_aggregates(toy_answers_df, by='worker'), expected_workers_accuracy)
    assert accuracy_on_aggregates(toy_answers_df) == 0.7083333333333334


def test_alpha_krippendorff(toy_answers_df: pd.DataFrame) -> None:
    assert alpha_krippendorff(pd.DataFrame.from_records([
        {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        {'task': 'Y', 'worker': 'A', 'label': 'No'},
        {'task': 'Y', 'worker': 'B', 'label': 'No'},
    ])) == 1.0

    assert alpha_krippendorff(pd.DataFrame.from_records([
        {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        {'task': 'Y', 'worker': 'A', 'label': 'No'},
        {'task': 'Y', 'worker': 'B', 'label': 'No'},
        {'task': 'Z', 'worker': 'A', 'label': 'Yes'},
        {'task': 'Z', 'worker': 'B', 'label': 'No'},
    ])) == 0.4444444444444444

    assert alpha_krippendorff(toy_answers_df) == 0.14219114219114215


def test_alpha_krippendorff_with_distance() -> None:
    whos_on_the_picture = pd.DataFrame.from_records([
        {'task': 'X', 'worker': 'A', 'label': frozenset(['dog'])},
        {'task': 'X', 'worker': 'B', 'label': frozenset(['dog'])},
        {'task': 'Y', 'worker': 'A', 'label': frozenset(['cat'])},
        {'task': 'Y', 'worker': 'B', 'label': frozenset(['cat'])},
        {'task': 'Z', 'worker': 'A', 'label': frozenset(['cat'])},
        {'task': 'Z', 'worker': 'B', 'label': frozenset(['cat', 'mouse'])},
    ])

    assert alpha_krippendorff(whos_on_the_picture) == 0.5454545454545454
    assert alpha_krippendorff(whos_on_the_picture, masi_distance) == 0.6673336668334168
