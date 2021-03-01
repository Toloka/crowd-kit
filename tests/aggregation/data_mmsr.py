import pytest
import pandas as pd
import numpy as np


# Wawa on toy YSDA

@pytest.fixture
def toy_labels_result_mmsr():
    return pd.DataFrame(
        [
            ['t1', 'yes'],
            ['t2', 'yes'],
            ['t3', 'no'],
            ['t4', 'yes'],
            ['t5', 'no'],
        ],
        columns=['task', 'label']
    )


@pytest.fixture
def toy_skills_result_mmsr():
    return pd.DataFrame(
        [
            ['w1', 0.328452],
            ['w2', 0.776393],
            ['w3', 0.759235],
            ['w4', 0.671548],
            ['w5', 0.776393],
        ],
        columns=['performer', 'skill']
    )


@pytest.fixture
def toy_probas_result_mmsr():
    result_df = pd.DataFrame(
        [
            [0.473821, 0.526179],
            [0.414814, 0.585186],
            [0.639077, 0.360923],
            [np.NaN, 1.000000],
            [0.840177, 0.159823],
        ],
        columns=['no', 'yes'],
        index=['t1', 't2', 't3', 't4', 't5'],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def simple_labels_result_mmsr(simple_ground_truth_df):
    return simple_ground_truth_df


@pytest.fixture
def simple_skills_result_mmsr():
    return pd.DataFrame(
        [
            ['0c3eb7d5fcc414db137c4180a654c06e', 0.210819],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', 0.789181],
            ['a452e450f913cfa987cad58d50393718', 0.789181],
            ['b17c3301ad2ccbb798716fdd405d16e8', 0.789181],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', 0.789181],
            ['e563e2fb32fce9f00123a65a1bc78c55', 0.779799],
        ],
        columns=['performer', 'skill']
    )


@pytest.fixture
def simple_probas_result_mmsr():
    result_df = pd.DataFrame(
        [
            [0.783892, np.NaN, 0.216108],
            [0.751367, 0.248633, np.NaN],
            [np.NaN, 0.844744, 0.155256],
            [np.NaN, np.NaN, 1.000000],
            [0.393067, 0.390206, 0.216726],
            [np.NaN, np.NaN, 1.000000],
            [np.NaN, 1.000000, np.NaN],
            [np.NaN, 0.783892, 0.216108],
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
