import pandas as pd

from crowdkit.metrics.data import consistency
from crowdkit.postprocessing import entropy_threshold
from crowdkit.aggregation import MajorityVote


class TestEntropyThreshold:
    def test_entropy_threshold_docstring_test(self):
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'performer': 'B', 'label': frozenset(['cat'])},
            ]
        )

        filtered_answers = entropy_threshold(answers)
        assert filtered_answers.columns.tolist() == ['task', 'performer', 'label']
        assert filtered_answers.shape == (3, 3)
        assert 'B' not in filtered_answers.performer

    def test_entropy_threshold_consistency_improves(self):
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'performer': 'C', 'label': frozenset(['cat'])},
                {'task': '1', 'performer': 'D', 'label': frozenset(['dog'])},

                {'task': '2', 'performer': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'C', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'D', 'label': frozenset(['dog'])},

                {'task': '3', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '3', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '3', 'performer': 'C', 'label': frozenset(['dog'])},
                {'task': '3', 'performer': 'D', 'label': frozenset(['dog'])},

                {'task': '4', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '4', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '4', 'performer': 'C', 'label': frozenset(['dog'])},
                {'task': '4', 'performer': 'D', 'label': frozenset(['dog'])},
            ]
        )

        skills = pd.Series(
            [1, 1, 1, 1],
            index=pd.Index(['A', 'B', 'C', 'D'], name='performer'),
        )

        base_consistency = consistency(answers, skills)
        filtered_answers = entropy_threshold(answers, skills, percentile=20)

        # B always answers "cat", his answers are useless and get filtered out
        assert 'B' not in filtered_answers.performer

        filtered_consistency = consistency(filtered_answers, skills)
        assert filtered_consistency > base_consistency

    def test_entropy_threshold_min_answers(self):
        answers = pd.DataFrame.from_records(
            [
                {'task': '1', 'performer': 'A', 'label': frozenset(['dog'])},
                {'task': '1', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '1', 'performer': 'C', 'label': frozenset(['cat'])},

                {'task': '2', 'performer': 'A', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'B', 'label': frozenset(['cat'])},
                {'task': '2', 'performer': 'D', 'label': frozenset(['dog'])},
            ]
        )

        filtered_answers = entropy_threshold(answers, min_answers=2)
        # B always answers "cat", his answers are useless and get filtered out
        assert 'B' not in filtered_answers.performer.values
        # C and D have one answer each and thus minimal entropy,
        # but left only 1 answer, so they don't get filtered out
        assert 'D' in filtered_answers.performer.values
        assert 'C' in filtered_answers.performer.values

        filtered_answers = entropy_threshold(answers, min_answers=1)
        assert 'B' not in filtered_answers.performer.values
        assert 'D' not in filtered_answers.performer.values
        assert 'C' not in filtered_answers.performer.values

    def test_entropy_threshold_simple_answers(self, simple_answers_df, simple_ground_truth_df):
        aggregated = MajorityVote().fit_predict(simple_answers_df)
        base_accuracy = sum(aggregated[simple_ground_truth_df.index] == simple_ground_truth_df)/len(simple_ground_truth_df)

        filtered_answers = entropy_threshold(simple_answers_df, percentile=20)
        assert 'e563e2fb32fce9f00123a65a1bc78c55' not in filtered_answers.performer.values
        assert '0c3eb7d5fcc414db137c4180a654c06e' not in filtered_answers.performer.values
        aggregated = MajorityVote().fit_predict(filtered_answers)
        filtered_accuracy = sum(aggregated[simple_ground_truth_df.index] == simple_ground_truth_df)/len(simple_ground_truth_df)

        assert filtered_accuracy >= base_accuracy