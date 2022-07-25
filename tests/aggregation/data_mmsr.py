import pandas as pd
import pytest


# Wawa on toy YSDA

@pytest.fixture
def toy_labels_result_mmsr() -> pd.Series:
    return pd.Series(
        ['yes', 'no', 'no', 'yes', 'no'],
        index=pd.Index(['t1', 't2', 't3', 't4', 't5'], name='task'),
        name='agg_label'
    )


@pytest.fixture
def toy_skills_result_mmsr() -> pd.Series:
    return pd.Series(
        [-0.9486439852160969, 0.9764628672747041, 1.2428113335479982, 0.948643985216097, 0.9764628672747041],
        pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='worker'),
        name='skill'
    )


@pytest.fixture
def toy_scores_result_mmsr() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [0.014244720916141606, 0.9857552790838584],
            [0.6023983861841676, 0.3976016138158324],
            [1.0, 3.474074997309249e-17],
            [0.0, 1.0],
            [1.2968466945188564, -0.2968466945188564],
        ],
        columns=['no', 'yes'],
        index=['t1', 't2', 't3', 't4', 't5'],
    )
    result_df.index.name = 'task'
    result_df.columns.name = 'label'
    return result_df


@pytest.fixture
def simple_labels_result_mmsr(simple_ground_truth: pd.Series) -> pd.Series:
    return simple_ground_truth


@pytest.fixture
def simple_skills_result_mmsr() -> pd.Series:
    skills = pd.Series({
        '0c3eb7d5fcc414db137c4180a654c06e': -0.6268515139467665,
        '0f65edea0a6dc7b9acba1dea313bbb3d': 2.0131458750666567,
        'a452e450f913cfa987cad58d50393718': 2.0131458750666567,
        'b17c3301ad2ccbb798716fdd405d16e8': 2.0131458750666567,
        'bde3b214b06c1efa6cb1bc6284dc72d2': 2.0131458750666567,
        'e563e2fb32fce9f00123a65a1bc78c55': 1.8527467194291514,
    }, name='skill')
    skills.index.name = 'worker'
    return skills


@pytest.fixture
def simple_scores_result_mmsr() -> pd.DataFrame:
    result_df = pd.DataFrame(
        [
            [1.1843984510373275, 0.0, -0.18439845103732766],
            [0.7652428480475112, 0.23475715195248875, 0.0],
            [0.0, 1.1158136796463138, -0.11581367964631392],
            [0.0, 0.0, 1.0],
            [0.6215252678195158, 0.5720046993399346, -0.19352996715945034],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1843984510373275, -0.18439845103732766],
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
