import pandas as pd
import pytest


# Gold Majority vote on toy YSDA

@pytest.fixture
def toy_labels_result_gold(toy_ground_truth_df: pd.Series) -> pd.Series:
    return toy_ground_truth_df


@pytest.fixture
def toy_skills_result_gold() -> pd.Series:
    return pd.Series(
        [0.5, 1.0, 1.0, 0.5, 0.0],
        pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
        name='skill'
    )


@pytest.fixture
def toy_probas_result_gold() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [0.750000, 0.250000],
            [0.833333, 0.166667],
            [0.333333, 0.666667],
            [1.0, 0.0],
            [0.166667, 0.833333],
        ],
        columns=['yes', 'no'],
        index=['t1', 't2', 't3', 't4', 't5']
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def toy_answers_on_gold_df_cannot_fit() -> pd.DataFrame:
    # When we have this dataset and 'toy_gold_df' we are trying to calculate the skills of the workers,
    # and we cannot do it for some workers
    return pd.DataFrame(
        [
            ['w1', 't1', 'no'],
            ['w2', 't2', 'yes'],
            ['w3', 't1', 'yes'],
            ['w4', 't2', 'yes'],
            ['w5', 't5', 'yes'],  # 'w5' answer, but 't5' not in 'toy_gold_df'
        ],
        columns=['worker', 'task', 'label']
    )


@pytest.fixture
def toy_answers_on_gold_df_cannot_predict() -> pd.DataFrame:
    # When we have this dataset in 'fit', and standart 'toy_answers_df' in predict and we cannot predict
    # labels or probas, because this dataset doesn't contain all workers from 'toy_answers_df'
    return pd.DataFrame(
        [
            ['w1', 't1', 'no'],
            ['w2', 't2', 'yes'],
            ['w3', 't1', 'yes'],
            ['w4', 't2', 'yes'],
            # ['w5', 't5', 'yes'],  # 'w5' missing here, but exists 'toy_answers_df'
        ],
        columns=['worker', 'task', 'label']
    )


# Gold Majority vote on simple

@pytest.fixture
def simple_labels_result_gold(simple_ground_truth: pd.Series) -> pd.Series:
    return simple_ground_truth


@pytest.fixture
def simple_skills_result_gold() -> pd.Series:
    skills = pd.Series({
        '0c3eb7d5fcc414db137c4180a654c06e': 0.5,
        '0f65edea0a6dc7b9acba1dea313bbb3d': 1.0,
        'a452e450f913cfa987cad58d50393718': 1.0,
        'b17c3301ad2ccbb798716fdd405d16e8': 1.0,
        'bde3b214b06c1efa6cb1bc6284dc72d2': 1.0,
        'e563e2fb32fce9f00123a65a1bc78c55': 0.5,
    }, name='skill')
    skills.index.name = 'worker'
    return skills


@pytest.fixture
def simple_probas_result_gold() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [0.8, 0.0, 0.2],
            [0.857143, 0.142857, 0.0],
            [0.0, 0.857143, 0.142857],
            [0.0, 0.0, 1.0],
            [0.5, 0.25, 0.25],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        columns=['chicken', 'goose', 'parrot'],
        index=[
            '1231239876--5fac0d234ffb2f3b00893ee4',
            '1231239876--5fac0d234ffb2f3b00893ee8',
            '1231239876--5fac0d234ffb2f3b00893eec',
            '1231239876--5fac0d234ffb2f3b00893efb',
            '1231239876--5fac0d234ffb2f3b00893efd',
            '1231239876--5fac0d234ffb2f3b00893f02',
            '1231239876--5fac0d234ffb2f3b00893f03',
            '1231239876--5fac0d234ffb2f3b00893f05',
            '1231239876--5fac0d234ffb2f3b00893f07',
            '1231239876--5fac0d234ffb2f3b00893f08',
        ],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def multiple_gt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['t1', 'w1', 'l1'],  # right
            ['t2', 'w1', 'l1'],
            ['t3', 'w1', 'l1'],  # wrong
            ['t1', 'w2', 'l2'],  # right
            ['t2', 'w2', 'l1'],
            ['t3', 'w2', 'l2'],  # right
            ['t1', 'w3', 'l3'],  # wrong
            ['t3', 'w3', 'l3'],  # wrong
        ],
        columns=['task', 'worker', 'label']
    )


@pytest.fixture
def multiple_gt_gt() -> pd.Series:
    return pd.Series(
        ['l1', 'l2', 'l2'],
        index=['t1', 't1', 't3']
    )


@pytest.fixture
def multiple_gt_aggregated() -> pd.Series:
    aggregated = pd.Series(
        ['l2', 'l1', 'l2'],
        index=['t1', 't2', 't3']
    )
    aggregated.index.name = 'task'
    aggregated.name = 'agg_label'
    return aggregated


@pytest.fixture
def multiple_gt_skills() -> pd.Series:
    skills = pd.Series(
        [0.5, 1., 0.],
        index=['w1', 'w2', 'w3'],
    )
    skills.index.name = 'worker'
    skills.name = 'skill'
    return skills
