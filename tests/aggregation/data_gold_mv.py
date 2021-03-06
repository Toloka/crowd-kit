import pandas as pd
import pytest


# Gold Majority vote on toy YSDA

@pytest.fixture
def toy_labels_result_gold(toy_ground_truth_df):
    return toy_ground_truth_df


@pytest.fixture
def toy_skills_result_gold():
    return pd.Series(
        [0.5, 1.0, 1.0, 0.5, 0.0],
        pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='performer'),
    )


@pytest.fixture
def toy_probas_result_gold():
    result_df = pd.DataFrame(
        [
            [0.750000, 0.250000],
            [0.833333, 0.166667],
            [0.333333, 0.666667],
            [1.0, 0.0],
            [0.166667, 0.833333],
        ],
        columns=['yes', 'no'],
        index=['t1', 't2', 't3', 't4', 't5'],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def toy_answers_on_gold_df_cannot_fit():
    # When we have this dataset and 'toy_gold_df' we are trying to calculate the skills of the performers,
    # and we cannot do it for some performers
    return pd.DataFrame(
        [
            ['w1', 't1', 'no'],
            ['w2', 't2', 'yes'],
            ['w3', 't1', 'yes'],
            ['w4', 't2', 'yes'],
            ['w5', 't5', 'yes'],  # 'w5' answer, but 't5' not in 'toy_gold_df'
        ],
        columns=['performer', 'task', 'label']
    )


@pytest.fixture
def toy_answers_on_gold_df_cannot_predict():
    # When we have this dataset in 'fit', and standart 'toy_answers_df' in predict and we cannot predict
    # labels or probas, because this dataset doesn't contain all performers from 'toy_answers_df'
    return pd.DataFrame(
        [
            ['w1', 't1', 'no'],
            ['w2', 't2', 'yes'],
            ['w3', 't1', 'yes'],
            ['w4', 't2', 'yes'],
            # ['w5', 't5', 'yes'],  # 'w5' missing here, but exists 'toy_answers_df'
        ],
        columns=['performer', 'task', 'label']
    )


# Gold Majority vote on simple

@pytest.fixture
def simple_labels_result_gold(simple_ground_truth_df):
    return simple_ground_truth_df


@pytest.fixture
def simple_skills_result_gold():
    skills = pd.Series({
        '0c3eb7d5fcc414db137c4180a654c06e': 0.5,
        '0f65edea0a6dc7b9acba1dea313bbb3d': 1.0,
        'a452e450f913cfa987cad58d50393718': 1.0,
        'b17c3301ad2ccbb798716fdd405d16e8': 1.0,
        'bde3b214b06c1efa6cb1bc6284dc72d2': 1.0,
        'e563e2fb32fce9f00123a65a1bc78c55': 0.5,
    })
    skills.index.name = 'performer'
    return skills


@pytest.fixture
def simple_probas_result_gold():
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
