import pandas as pd
import pytest


# Wawa on toy YSDA

@pytest.fixture
def toy_labels_result_wawa() -> pd.Series:
    return pd.Series(
        ['no', 'yes', 'no', 'yes', 'no'],
        pd.Index(['t1', 't2', 't3', 't4', 't5'], name='task'),
        name='agg_label'
    )


@pytest.fixture
def toy_skills_result_wawa() -> pd.Series:
    return pd.Series(
        [0.6, 0.8, 1.0, 0.4, 0.8],
        pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
        name='skill'
    )


@pytest.fixture
def toy_probas_result_wawa() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [0.461538, 0.538462],
            [0.666667, 0.333333],
            [0.277778, 0.722222],
            [1.0, 0],
            [0.166667, 0.833333],
        ],
        columns=['yes', 'no'],
        index=['t1', 't2', 't3', 't4', 't5'],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


# Wawa on simple

@pytest.fixture
def simple_labels_result_wawa(simple_ground_truth: pd.Series) -> pd.Series:
    return simple_ground_truth


@pytest.fixture
def simple_skills_result_wawa() -> pd.Series:
    skills = pd.Series({
        '0c3eb7d5fcc414db137c4180a654c06e': 1/3,
        '0f65edea0a6dc7b9acba1dea313bbb3d': 1.0,
        'a452e450f913cfa987cad58d50393718': 1.0,
        'b17c3301ad2ccbb798716fdd405d16e8': 1.0,
        'bde3b214b06c1efa6cb1bc6284dc72d2': 1.0,
        'e563e2fb32fce9f00123a65a1bc78c55': 2/3,
    }, name='skill')
    skills.index.name = 'worker'
    return skills


@pytest.fixture
def simple_probas_result_wawa() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [0.857143, 0, 0.142857],  # [9/11, 0.0, 2/11] ?
            [0.818182, 0.181818, 0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.500000, 0.333333, 0.166667],
            [0.0, 0.0, 1.0],
            [0.0, 1.000000, 0.0],
            [0.0, 0.857143, 0.142857],
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
