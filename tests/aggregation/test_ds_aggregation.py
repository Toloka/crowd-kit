"""
Simplest aggregation algorithms tests on toy YSDA dataset
Testing all boundary conditions and asserts
"""

from typing import Any, List, Literal, Optional, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from crowdkit.aggregation import DawidSkene, OneCoinDawidSkene


class TestWorkerInitError:

    @pytest.mark.parametrize(
        "n_iter, tol, strategy",
        [
            (10, 0, "addition"),
            (10, 0, "assign"),
            (10, 0, None),
        ],
    )
    def test_without_initial_error_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Optional[Literal["assign", "addition"]],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
    ) -> None:
        """
        Basic parameter compatibility test: None parameters
        """
        np.random.seed(42)
        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        assert_series_equal(
            ds.fit(toy_answers_df, initial_error=None).labels_.sort_index(),  # type: ignore
            toy_ground_truth_df.sort_index(),
        )

    @pytest.mark.parametrize(
        "n_iter, tol, strategy",
        [
            (10, 0, "addition"),
            (10, 0, "assign"),
            (100500, 1e-5, "addition"),
            (100500, 1e-5, "assign"),
        ],
    )
    def test_zero_error_addition_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: pd.DataFrame,
    ) -> None:
        """
        Basic parameter compatibility test: zeros initial error matrix
        """
        np.random.seed(42)
        initial_error_df = toy_worker_init_error_zero_df
        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        if strategy == "addition":
            assert_series_equal(
                ds.fit(toy_answers_df, initial_error=initial_error_df).labels_.sort_index(),  # type: ignore
                toy_ground_truth_df.sort_index(),
            )
        else:
            with pytest.raises(
                ValueError,
                match="The sum of each worker's error matrix in initial_error should be 1.0",
            ):
                ds.fit(toy_answers_df, initial_error=initial_error_df)

    @pytest.mark.parametrize(
        "n_iter, tol, strategy",
        [
            (10, 0, "addition"),
            (10, 0, "assign"),
            (100500, 1e-5, "addition"),
            (100500, 1e-5, "assign"),
        ],
    )
    def test_zero_partial_error_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: pd.DataFrame,
    ) -> None:
        """
        Basic parameter compatibility test: when initial_error doesn't contain all workers
        """
        np.random.seed(42)
        initial_error_df = toy_worker_init_error_zero_df[:3]
        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        if strategy == "addition":
            assert_series_equal(
                ds.fit(toy_answers_df, initial_error=initial_error_df).labels_.sort_index(),  # type: ignore
                toy_ground_truth_df.sort_index(),
            )
        else:
            with pytest.raises(
                ValueError,
            ):
                ds.fit(toy_answers_df, initial_error=initial_error_df)

    @pytest.mark.parametrize(
        "n_iter, tol, strategy", [(10, 0, "addition"), (100500, 1e-5, "addition")]
    )
    def test_addition_consistency_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Behavior test: when worker's init error matrix is similar to the error matrix in these tasks,
        the aggregation result should be the same as before(without init error)
        """
        np.random.seed(42)
        # According to the ground truth data, w2's answer is always right.
        # so when we set the initial error matrix of w2 to almost right, we should get same results
        init_error_df = toy_worker_init_error_zero_df
        init_error_df[("w2", "yes"), "no"] = 1
        init_error_df[("w2", "yes"), "yes"] = 99
        init_error_df[("w2", "no"), "yes"] = 1
        init_error_df[("w2", "no"), "no"] = 99

        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        assert_series_equal(
            ds.fit(toy_answers_df, initial_error=init_error_df).labels_.sort_index(),  # type: ignore
            toy_ground_truth_df.sort_index(),
        )

    @pytest.mark.parametrize(
        "n_iter, tol, strategy", [(10, 0, "assign"), (100500, 1e-5, "assign")]
    )
    def test_assign_consistency_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Behavior test: when worker's init error matrix is similar to the error matrix in these tasks,
        the aggregation result should be the same as before(without init error)
        """
        # step1: get the original estimated error matrix
        np.random.seed(42)
        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=None)
        assert_series_equal(
            ds.fit(toy_answers_df, initial_error=None).labels_.sort_index(),  # type: ignore
            toy_ground_truth_df.sort_index(),
        )
        original_error_df = ds.errors_

        # step2: use the original_error_df as initial_error to fit the model
        init_error_df = original_error_df
        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        # step3: check the result, which should be the same as the original one
        assert_series_equal(
            ds.fit(toy_answers_df, initial_error=init_error_df).labels_.sort_index(),  # type: ignore
            toy_ground_truth_df.sort_index(),
        )

    @pytest.mark.parametrize(
        "n_iter, tol, strategy", [(10, 0, "addition"), (100500, 1e-5, "addition")]
    )
    def test_addition_desired_label_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Behavior test: dedicate init error matrices should lead to desired results
        """
        np.random.seed(42)
        # worker's annotation on t2: w1: yes, w2: yes, w3: yes, w4: no, w5: no
        # ground truth: t2: yes

        # When we set workers' init error matrices as fellow, we should get the desired result
        # In these case, we want the t2's label to be no rather than yes
        init_error_df = toy_worker_init_error_zero_df
        item_indexes = [
            [("w1", "yes"), "no"],
            [("w2", "yes"), "no"],
            [("w3", "yes"), "no"],
            [("w4", "no"), "no"],
            [("w5", "no"), "no"],
        ]
        for loc in item_indexes:
            init_error_df.loc[loc[0], loc[1]] = 99  # type: ignore

        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        ds = ds.fit(toy_answers_df, initial_error=init_error_df)  # type: ignore
        assert ds.labels_["t2"] == "no"  # type: ignore

    @pytest.mark.parametrize(
        "n_iter, tol, strategy", [(10, 0, "assign"), (100500, 1e-5, "assign")]
    )
    def test_assign_desired_label_on_toy_ysda(
        self,
        n_iter: int,
        tol: float,
        strategy: Literal["assign", "addition"],
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Behavior test: dedicate init error matrices should lead to desired results
        """
        np.random.seed(42)
        # worker's annotation on t2: w1: yes, w2: yes, w3: yes, w4: no, w5: no
        # ground truth: t2: yes

        # When we set workers' init error matrices as fellow, we should get the desired result
        # In this case, we want the t2's label to be no rather than yes
        # init all probability with 0.5
        init_error_df = toy_worker_init_error_zero_df
        init_error_df.loc[:, :] = 0.5
        # set dedicated probability
        item_indexes = [
            [("w1", "yes"), ("w1", "no")],
            [("w2", "yes"), ("w2", "no")],
            [("w3", "yes"), ("w3", "no")],
            [("w4", "no"), ("w4", "yes")],
            [("w5", "no"), ("w5", "yes")],
        ]
        for loc in item_indexes:
            init_error_df.loc[loc[0], "no"] = 0.99
            init_error_df.loc[loc[1], "no"] = 0.01

        ds = DawidSkene(n_iter=n_iter, tol=tol, initial_error_strategy=strategy)
        ds = ds.fit(toy_answers_df, initial_error=init_error_df)  # type: ignore
        assert ds.labels_["t2"] == "no"  # type: ignore

    def test_addition_inner_state_on_toy_ysda(
        self,
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Inner state test.

        Without init error, w2's error matrix(without avg) in iter 0 is as fellows:

                no  yes
        label
        yes    0.9  2.1
        no     1.4  0.6

        After fitting with init error, w2's error matrix(without avg) in iter 0 should be:

                no  yes
        label
        yes    2  3
        no     2  1

        After the avg of error matrix:
                no  yes
        label
        yes    0.5  0.75
        no     0.5  0.25
        """
        np.random.seed(42)
        init_error_df = toy_worker_init_error_zero_df
        init_error_df.loc[("w2", "yes"), "no"] = 1.1  # 1.1 + 0.9 = 2
        init_error_df.loc[("w2", "yes"), "yes"] = 0.9  # 0.9 + 2.1 = 3
        init_error_df.loc[("w2", "no"), "yes"] = 0.4  # 0.4 + 0.6 = 1
        init_error_df.loc[("w2", "no"), "no"] = 0.6  # 0.6 + 1.4 = 2

        # fit with init error
        with_init_errors = DawidSkene(
            n_iter=0,
            tol=0.0,
            initial_error_strategy="addition",
        ).fit(
            toy_answers_df, initial_error=init_error_df  # type: ignore
        )

        # check w2 error matrix
        item_probs = [
            ("yes", "no", 0.5),
            ("yes", "yes", 0.75),
            ("no", "yes", 0.25),
            ("no", "no", 0.5),
        ]
        for observed, label, prob in item_probs:
            assert np.isclose(with_init_errors.errors_.loc["w2"].loc[observed, label], prob)  # type: ignore

    def test_assign_inner_state_on_toy_ysda(
        self,
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: "pd.Series[Any]",
        toy_worker_init_error_zero_df: "pd.Series[Any]",
    ) -> None:
        """
        Inner state test.
        """
        np.random.seed(42)
        # generate random init error matrix
        init_error_df = toy_worker_init_error_zero_df
        init_error_df.loc[:, :] = np.random.randint(1, 100, size=init_error_df.shape)
        init_error_df = (
            init_error_df / init_error_df.groupby("worker", sort=False).sum()
        )
        # fit with init error
        ds = DawidSkene(
            n_iter=0,
            tol=0.0,
            initial_error_strategy="assign",
        ).fit(
            toy_answers_df, initial_error=init_error_df  # type: ignore
        )

        # check w2 error matrix
        assert_frame_equal(init_error_df, ds.errors_)  # type: ignore


@pytest.mark.parametrize("n_iter, tol", [(10, 0), (100500, 1e-5)])
def test_aggregate_ds_gold_on_toy_ysda(
    n_iter: int,
    tol: float,
    toy_answers_df: pd.DataFrame,
    toy_ground_truth_df: "pd.Series[Any]",
    toy_gold_df: "pd.Series[Any]",
) -> None:
    np.random.seed(42)
    assert_series_equal(
        DawidSkene(n_iter=n_iter, tol=tol).fit(toy_answers_df, toy_gold_df).labels_.sort_index(),  # type: ignore
        toy_ground_truth_df.sort_index(),
    )


@pytest.mark.parametrize("n_iter", [0, 1, 2])
def test_ds_gold_probas_correction_with_iters(
    n_iter: int,
    toy_answers_df: pd.DataFrame,
    toy_ground_truth_df: "pd.Series[Any]",
    toy_gold_df: "pd.Series[Any]",
) -> None:
    ds = DawidSkene(n_iter).fit(toy_answers_df, toy_gold_df)
    probas = ds.probas_
    assert probas is not None, "no probas_"
    probas = probas.merge(
        toy_gold_df.rename("true_label"), left_on="task", right_index=True
    )
    # check that gold label probas are correct, i.e. equal to 1.0
    match_count = probas.apply(
        lambda row: np.isclose(row[row.true_label], 1.0, atol=1e-8), axis=1
    ).sum()
    assert match_count == len(toy_gold_df), f"{match_count=}, {len(toy_gold_df)=}"
    # check that all probas sum to 1(check that all probas are correct)
    assert np.allclose(probas.drop("true_label", axis=1).sum(axis=1), 1.0, atol=1e-8)
    # check labels
    assert ds.labels_ is not None, "no labels_"
    assert_series_equal(ds.labels_[toy_gold_df.index], toy_gold_df, check_names=False)


@pytest.mark.parametrize("n_iter, tol", [(10, 0), (100500, 1e-5)])
def test_aggregate_ds_on_toy_ysda(
    n_iter: int,
    tol: float,
    toy_answers_df: pd.DataFrame,
    toy_ground_truth_df: "pd.Series[Any]",
) -> None:
    np.random.seed(42)
    assert_series_equal(
        DawidSkene(n_iter=n_iter, tol=tol).fit(toy_answers_df).labels_.sort_index(),  # type: ignore
        toy_ground_truth_df.sort_index(),
    )


@pytest.mark.parametrize("n_iter, tol", [(10, 0), (100500, 1e-5)])
def test_aggregate_hds_on_toy_ysda(
    n_iter: int,
    tol: float,
    toy_answers_df: pd.DataFrame,
    toy_ground_truth_df: "pd.Series[Any]",
) -> None:
    np.random.seed(42)
    assert_series_equal(
        OneCoinDawidSkene(n_iter=n_iter, tol=tol).fit(toy_answers_df).labels_.sort_index(),  # type: ignore
        toy_ground_truth_df.sort_index(),
    )

    assert_series_equal(
        OneCoinDawidSkene(n_iter=n_iter, tol=tol)
        .fit_predict(toy_answers_df)
        .sort_index(),
        toy_ground_truth_df.sort_index(),
    )

    probas = OneCoinDawidSkene(n_iter=n_iter, tol=tol).fit_predict_proba(toy_answers_df)
    assert ((probas >= 0) & (probas <= 1)).all().all()


@pytest.mark.parametrize("n_iter, tol", [(10, 0), (100500, 1e-5)])
def test_aggregate_ds_on_simple(
    n_iter: int,
    tol: float,
    simple_answers_df: pd.DataFrame,
    simple_ground_truth: "pd.Series[Any]",
) -> None:
    np.random.seed(42)
    assert_series_equal(
        DawidSkene(n_iter=n_iter, tol=tol).fit(simple_answers_df).labels_.sort_index(),  # type: ignore
        simple_ground_truth.sort_index(),
    )


@pytest.mark.parametrize("n_iter, tol", [(10, 0), (100500, 1e-5)])
def test_aggregate_hds_on_simple(
    n_iter: int,
    tol: float,
    simple_answers_df: pd.DataFrame,
    simple_ground_truth: "pd.Series[Any]",
) -> None:
    np.random.seed(42)
    assert_series_equal(
        OneCoinDawidSkene(n_iter=n_iter, tol=tol).fit(simple_answers_df).labels_.sort_index(),  # type: ignore
        simple_ground_truth.sort_index(),
    )

    assert_series_equal(
        OneCoinDawidSkene(n_iter=n_iter, tol=tol)
        .fit_predict(simple_answers_df)
        .sort_index(),
        simple_ground_truth.sort_index(),
    )

    probas = OneCoinDawidSkene(n_iter=n_iter, tol=tol).fit_predict_proba(
        simple_answers_df
    )
    assert ((probas >= 0) & (probas <= 1)).all().all()


def _make_probas(data: List[List[Any]]) -> pd.DataFrame:
    # TODO: column should not be an index!
    columns = pd.Index(["task", "no", "yes"], name="label")
    return pd.DataFrame(data, columns=columns).set_index("task")


def _make_tasks_labels(data: List[List[Any]]) -> pd.DataFrame:
    # TODO: should task be indexed?
    return cast(
        pd.DataFrame,
        pd.DataFrame(data, columns=["task", "label"])
        .set_index("task")
        .squeeze()
        .rename("agg_label"),
    )


def _make_errors(data: List[List[Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        data,
        columns=["worker", "label", "no", "yes"],
    ).set_index(["worker", "label"])


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["t1", "w1", "no"],
            ["t1", "w2", "yes"],
            # ['t1', 'w3', np.NaN],
            ["t1", "w4", "yes"],
            ["t1", "w5", "no"],
            ["t2", "w1", "yes"],
            ["t2", "w2", "yes"],
            ["t2", "w3", "yes"],
            ["t2", "w4", "no"],
            ["t2", "w5", "no"],
            ["t3", "w1", "yes"],
            ["t3", "w2", "no"],
            ["t3", "w3", "no"],
            ["t3", "w4", "yes"],
            ["t3", "w5", "no"],
            ["t4", "w1", "yes"],
            ["t4", "w2", "yes"],
            ["t4", "w3", "yes"],
            ["t4", "w4", "yes"],
            ["t4", "w5", "yes"],
            ["t5", "w1", "yes"],
            ["t5", "w2", "no"],
            ["t5", "w3", "no"],
            ["t5", "w4", "no"],
            ["t5", "w5", "no"],
        ],
        columns=["task", "worker", "label"],
    )


@pytest.fixture
def probas_iter_0() -> pd.DataFrame:
    return _make_probas(
        [
            ["t1", 0.5, 0.5],
            ["t2", 0.4, 0.6],
            ["t3", 0.6, 0.4],
            ["t4", 0.0, 1.0],
            ["t5", 0.8, 0.2],
        ]
    )


@pytest.fixture
def priors_iter_0() -> "pd.Series[Any]":
    return pd.Series([0.46, 0.54], pd.Index(["no", "yes"], name="label"), name="prior")


@pytest.fixture
def tasks_labels_iter_0() -> pd.DataFrame:
    return _make_tasks_labels(
        [
            ["t1", "no"],
            ["t2", "yes"],
            ["t3", "no"],
            ["t4", "yes"],
            ["t5", "no"],
        ]
    )


@pytest.fixture
def errors_iter_0() -> pd.DataFrame:
    return _make_errors(
        [
            ["w1", "no", 0.22, 0.19],
            ["w1", "yes", 0.78, 0.81],
            ["w2", "no", 0.61, 0.22],
            ["w2", "yes", 0.39, 0.78],
            ["w3", "no", 0.78, 0.27],
            ["w3", "yes", 0.22, 0.73],
            ["w4", "no", 0.52, 0.30],
            ["w4", "yes", 0.48, 0.70],
            ["w5", "no", 1.00, 0.63],
            ["w5", "yes", 0.00, 0.37],
        ]
    )


@pytest.fixture
def probas_iter_1() -> pd.DataFrame:
    return _make_probas(
        [
            ["t1", 0.35, 0.65],
            ["t2", 0.26, 0.74],
            ["t3", 0.87, 0.13],
            ["t4", 0.00, 1.00],
            ["t5", 0.95, 0.05],
        ]
    )


@pytest.fixture
def priors_iter_1() -> "pd.Series[Any]":
    return pd.Series([0.49, 0.51], pd.Index(["no", "yes"], name="label"), name="prior")


@pytest.fixture
def tasks_labels_iter_1() -> pd.DataFrame:
    return _make_tasks_labels(
        [
            ["t1", "yes"],
            ["t2", "yes"],
            ["t3", "no"],
            ["t4", "yes"],
            ["t5", "no"],
        ]
    )


@pytest.fixture
def errors_iter_1() -> pd.DataFrame:
    return _make_errors(
        [
            ["w1", "no", 0.14, 0.25],
            ["w1", "yes", 0.86, 0.75],
            ["w2", "no", 0.75, 0.07],
            ["w2", "yes", 0.25, 0.93],
            ["w3", "no", 0.87, 0.09],
            ["w3", "yes", 0.13, 0.91],
            ["w4", "no", 0.50, 0.31],
            ["w4", "yes", 0.50, 0.69],
            ["w5", "no", 1.00, 0.61],
            ["w5", "yes", 0.00, 0.39],
        ]
    )


@pytest.mark.parametrize("n_iter", [0, 1])
def test_dawid_skene_step_by_step(
    request: Any, data: pd.DataFrame, n_iter: int
) -> None:
    probas = request.getfixturevalue(f"probas_iter_{n_iter}")
    labels = request.getfixturevalue(f"tasks_labels_iter_{n_iter}")
    errors = request.getfixturevalue(f"errors_iter_{n_iter}")
    priors = request.getfixturevalue(f"priors_iter_{n_iter}")

    ds = DawidSkene(n_iter).fit(data)
    assert ds.probas_ is not None, "no probas_"
    assert ds.errors_ is not None, "no errors_"
    assert ds.priors_ is not None, "no priors_"
    assert ds.labels_ is not None, "no labels_"
    assert_frame_equal(probas, ds.probas_, check_like=True, atol=0.005)
    assert_frame_equal(errors, ds.errors_, check_like=True, atol=0.005)
    assert_series_equal(priors, ds.priors_, atol=0.005)
    assert_series_equal(labels, ds.labels_, atol=0.005)


def test_dawid_skene_on_empty_input(request: Any, data: pd.DataFrame) -> None:
    ds = DawidSkene(10).fit(pd.DataFrame([], columns=["task", "worker", "label"]))
    assert ds.probas_ is not None, "no probas_"
    assert ds.errors_ is not None, "no errors_"
    assert ds.priors_ is not None, "no priors_"
    assert ds.labels_ is not None, "no labels_"
    assert_frame_equal(pd.DataFrame(), ds.probas_, check_like=True, atol=0.005)
    assert_frame_equal(pd.DataFrame(), ds.errors_, check_like=True, atol=0.005)
    assert_series_equal(pd.Series(dtype=float, name="prior"), ds.priors_, atol=0.005)
    assert_series_equal(
        pd.Series(dtype=float, name="agg_label"), ds.labels_, atol=0.005
    )


@pytest.mark.parametrize("overlap", [3, 300, 30000])
def test_dawid_skene_overlap(overlap: int) -> None:
    data = pd.DataFrame(
        [
            {
                "task": task_id,
                "worker": perf_id,
                "label": "yes" if (perf_id - task_id) % 3 else "no",
            }
            for perf_id in range(overlap)
            for task_id in range(3)
        ]
    )

    ds = DawidSkene(20).fit(data)

    expected_probas = _make_probas([[task_id, 1 / 3.0, 2 / 3] for task_id in range(3)])
    expected_labels = _make_tasks_labels([[task_id, "yes"] for task_id in range(3)])

    # TODO: check errors_
    assert ds.probas_ is not None, "no probas_"
    assert ds.errors_ is not None, "no errors_"
    assert ds.priors_ is not None, "no priors_"
    assert ds.labels_ is not None, "no labels_"
    assert_frame_equal(expected_probas, ds.probas_, check_like=True, atol=0.005)
    assert_series_equal(expected_labels, ds.labels_, atol=0.005)  # type: ignore
    assert_series_equal(
        pd.Series([1 / 3, 2 / 3], pd.Index(["no", "yes"], name="label"), name="prior"),
        ds.priors_,
        atol=0.005,
    )


def test_ds_on_bool_labels(
    data_with_bool_labels: pd.DataFrame, bool_labels_ground_truth: "pd.Series[Any]"
) -> None:
    ds = DawidSkene(20).fit(data_with_bool_labels)
    assert ds.labels_ is not None, "no labels_"
    assert_series_equal(bool_labels_ground_truth, ds.labels_, atol=0.005)


def test_hds_on_bool_labels(
    data_with_bool_labels: pd.DataFrame, bool_labels_ground_truth: "pd.Series[Any]"
) -> None:
    hds = OneCoinDawidSkene(20).fit(data_with_bool_labels)
    assert hds.labels_ is not None, "no labels_"
    assert_series_equal(bool_labels_ground_truth, hds.labels_, atol=0.005)
