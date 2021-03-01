import pytest
import pandas as pd
import numpy as np


# Wawa on toy YSDA

@pytest.fixture
def toy_labels_result_zbs():
    return pd.DataFrame(
        [
            ['t1', 'no'],
            ['t2', 'yes'],
            ['t3', 'no'],
            ['t4', 'yes'],
            ['t5', 'no'],
        ],
        columns=['task', 'label']
    )


@pytest.fixture
def toy_skills_result_zbs():
    return pd.DataFrame(
        [
            ['w1', 0.6],
            ['w2', 0.8],
            ['w3', 1.0],
            ['w4', 0.4],
            ['w5', 0.8],
        ],
        columns=['performer', 'skill']
    )


@pytest.fixture
def toy_probas_result_zbs():
    result_df = pd.DataFrame(
        [
            [0.538462, 0.461538],
            [0.333333, 0.666667],
            [0.722222, 0.277778],
            [np.NaN, 1.000000],
            [0.833333, 0.166667],
        ],
        columns=['no', 'yes'],
        index=['t1', 't2', 't3', 't4', 't5'],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def simple_labels_result_zbs(simple_ground_truth_df):
    return simple_ground_truth_df


@pytest.fixture
def simple_skills_result_zbs():
    return pd.DataFrame(
        [
            ['0c3eb7d5fcc414db137c4180a654c06e', 0.333333],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', 1.000000],
            ['a452e450f913cfa987cad58d50393718', 1.000000],
            ['b17c3301ad2ccbb798716fdd405d16e8', 1.000000],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', 1.000000],
            ['e563e2fb32fce9f00123a65a1bc78c55', 0.666667],
        ],
        columns=['performer', 'skill']
    )


@pytest.fixture
def simple_probas_result_zbs():
    result_df = pd.DataFrame(
        [
            [0.857143, np.NaN, 0.142857],
            [0.818182, 0.181818, np.NaN],
            [np.NaN, 0.900000, 0.100000],
            [np.NaN, np.NaN, 1.000000],
            [0.500000, 0.333333, 0.166667],
            [np.NaN, np.NaN, 1.000000],
            [np.NaN, 1.000000, np.NaN],
            [np.NaN, 0.857143, 0.142857],
            [np.NaN, np.NaN, 1.000000],
            [np.NaN, np.NaN, 1.000000],
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
