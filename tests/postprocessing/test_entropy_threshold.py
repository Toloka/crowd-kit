import pandas as pd
import pytest

from crowdkit.metrics.data import consistency
from crowdkit.postprocessing import entropy_threshold
from crowdkit.aggregation import MajorityVote


class TestEntropyThreshold:
    def test_entropy_threshold_docstring_test(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'worker': 'B', 'label': frozenset(['cat'])},
            ]
        )
        with pytest.warns(UserWarning, match='Removed >= 1/2 of answers with entropy_threshold. This might lead to poor annotation quality. '):
            filtered_answers = entropy_threshold(answers)
        assert filtered_answers.columns.tolist() == ['task', 'worker', 'label']
        assert filtered_answers.shape == (3, 3)
        assert 'B' not in filtered_answers.worker

    def test_entropy_threshold_consistency_improves(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'D', 'label': frozenset(['dog'])},

                {'task': '2', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'C', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'D', 'label': frozenset(['dog'])},

                {'task': '3', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'worker': 'C', 'label': frozenset(['dog'])},
                {'task': '3', 'worker': 'D', 'label': frozenset(['dog'])},

                {'task': '4', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '4', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '4', 'worker': 'C', 'label': frozenset(['dog'])},
                {'task': '4', 'worker': 'D', 'label': frozenset(['dog'])},
            ]
        )

        skills = pd.Series(
            [1, 1, 1, 1],
            index=pd.Index(['A', 'B', 'C', 'D'], name='worker'),
        )

        base_consistency = consistency(answers, skills)
        with pytest.warns(UserWarning, match='Removed >= 1/2 of answers with entropy_threshold. This might lead to poor annotation quality. '):
            filtered_answers = entropy_threshold(answers, skills, percentile=20)

        # B always answers "cat", his answers are useless and get filtered out
        assert 'B' not in filtered_answers.worker

        filtered_consistency = consistency(filtered_answers, skills)
        assert filtered_consistency > base_consistency

    def test_entropy_threshold_min_answers(self) -> None:
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'worker': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'worker': 'C', 'label': frozenset(['cat'])},

                {'task': '2', 'worker': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'worker': 'D', 'label': frozenset(['dog'])},
            ]
        )

        filtered_answers = entropy_threshold(answers, min_answers=2)
        # B always answers "cat", his answers are useless and get filtered out
        assert 'B' not in filtered_answers.worker.values
        # C and D have one answer each and thus minimal entropy,
        # but left only 1 answer, so they don't get filtered out
        assert 'D' in filtered_answers.worker.values
        assert 'C' in filtered_answers.worker.values
        with pytest.warns(UserWarning, match='Removed >= 1/2 of answers with entropy_threshold. This might lead to poor annotation quality. '):
            filtered_answers = entropy_threshold(answers, min_answers=1)
        assert 'B' not in filtered_answers.worker.values
        assert 'D' not in filtered_answers.worker.values
        assert 'C' not in filtered_answers.worker.values

    def test_entropy_threshold_simple_answers(self, simple_answers_df: pd.DataFrame, simple_ground_truth: pd.Series) -> None:
        aggregated = MajorityVote().fit_predict(simple_answers_df)
        base_accuracy = sum(aggregated[simple_ground_truth.index] == simple_ground_truth)/len(simple_ground_truth)

        filtered_answers = entropy_threshold(simple_answers_df, percentile=20)
        assert 'e563e2fb32fce9f00123a65a1bc78c55' not in filtered_answers.worker.values
        assert '0c3eb7d5fcc414db137c4180a654c06e' not in filtered_answers.worker.values
        aggregated = MajorityVote().fit_predict(filtered_answers)
        filtered_accuracy = sum(aggregated[simple_ground_truth.index] == simple_ground_truth)/len(simple_ground_truth)

        assert filtered_accuracy >= base_accuracy
