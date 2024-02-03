__all__ = ["DawidSkene", "OneCoinDawidSkene"]

from typing import Any, List, Optional, cast

import attr
import numpy as np
import pandas as pd

from ..base import BaseClassificationAggregator
from ..utils import get_most_probable_labels, named_series_attrib
from .majority_vote import MajorityVote

_EPS = np.float_power(10, -10)


@attr.s
class DawidSkene(BaseClassificationAggregator):
    r"""The **Dawid-Skene** aggregation model is a probabilistic model that parametrizes the expertise level of workers with confusion matrices.

    Let $e^w$ be a worker confusion (error) matrix of size $K \times K$ in case of the $K$ class classification,
    $p$ be a vector of prior class probabilities, $z_j$ be a true task label, and $y^w_j$ be a worker
    response to the task $j$. The relationship between these parameters is represented by the following latent
    label model.

    ![Dawid-Skene latent label model](https://tlk.s3.yandex.net/crowd-kit/docs/ds_llm.png)

    Here the prior true label probability is
    $$
    \operatorname{Pr}(z_j = c) = p[c],
    $$
    and the probability distribution of the worker responses with the true label $c$ is represented by the
    corresponding column of the error matrix:
    $$
    \operatorname{Pr}(y_j^w = k | z_j = c) = e^w[k, c].
    $$

    Parameters $p$, $e^w$, and latent variables $z$ are optimized with the Expectation-Maximization algorithm:
    1. **E-step**. Estimates the true task label probabilities using the specified workers' responses,
        the prior label probabilities, and the workers' error probability matrix.
    2. **M-step**. Estimates the workers' error probability matrix using the specified workers' responses and the true task label probabilities.

    A. Philip Dawid and Allan M. Skene. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    *Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28*, 1 (1979), 20–28.

    <https://doi.org/10.2307/2346806>

    Examples:
        >>> from crowdkit.aggregation import DawidSkene
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> ds = DawidSkene(100)
        >>> result = ds.fit_predict(df)
    """

    n_iter: int = attr.ib(default=100)
    """The maximum number of EM iterations."""

    tol: float = attr.ib(default=1e-5)
    """The tolerance stopping criterion for iterative methods with a variable number of steps.
    The algorithm converges when the loss change is less than the `tol` parameter."""

    probas_: Optional[pd.DataFrame] = attr.ib(init=False)
    """The probability distributions of task labels.
    The `pandas.Series` data is indexed by `task` so that `labels.loc[task]` is the most likely true label of tasks."""

    priors_: Optional["pd.Series[Any]"] = named_series_attrib(name="prior")
    """The prior label distribution.
    The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that
    the `task` true label is equal to `label`. Each probability is in the range from 0 to 1, all task probabilities
    must sum up to 1."""

    errors_: Optional[pd.DataFrame] = attr.ib(init=False)
    """The workers' error matrices. The `pandas.DataFrame` data is indexed by `worker` and `label` with a column
    for every `label_id` found in `data` so that `result.loc[worker, observed_label, true_label]` is the probability
    that `worker` produces `observed_label`, given that the task true label is `true_label`."""

    loss_history_: List[float] = attr.ib(init=False)
    """ A list of loss values during training."""

    @staticmethod
    def _m_step(data: pd.DataFrame, probas: pd.DataFrame) -> pd.DataFrame:
        """Performs M-step of the Dawid-Skene algorithm.

        Estimates the workers' error probability matrix using the specified workers' responses and the true task label probabilities.
        """
        joined = data.join(probas, on="task")
        joined.drop(columns=["task"], inplace=True)

        errors = joined.groupby(["worker", "label"], sort=False).sum()
        errors.clip(lower=_EPS, inplace=True)
        errors /= errors.groupby("worker", sort=False).sum()

        return errors

    @staticmethod
    def _e_step(
        data: pd.DataFrame, priors: "pd.Series[Any]", errors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs E-step of the Dawid-Skene algorithm.

        Estimates the true task label probabilities using the specified workers' responses,
        the prior label probabilities, and the workers' error probability matrix.
        """

        # We have to multiply lots of probabilities and such products are known to converge
        # to zero exponentially fast. To avoid floating-point precision problems we work with
        # logs of original values
        joined = data.join(np.log2(errors), on=["worker", "label"])  # type: ignore
        joined.drop(columns=["worker", "label"], inplace=True)
        log_likelihoods = np.log2(priors) + joined.groupby("task", sort=False).sum()
        log_likelihoods.rename_axis("label", axis=1, inplace=True)

        # Exponentiating log_likelihoods 'as is' may still get us beyond our precision.
        # So we shift every row of log_likelihoods by a constant (which is equivalent to
        # multiplying likelihoods rows by a constant) so that max log_likelihood in each
        # row is equal to 0. This trick ensures proper scaling after exponentiating and
        # does not affect the result of E-step
        scaled_likelihoods = np.exp2(
            log_likelihoods.sub(log_likelihoods.max(axis=1), axis=0)
        )
        scaled_likelihoods = scaled_likelihoods.div(
            scaled_likelihoods.sum(axis=1), axis=0
        )
        # Convert columns types to label type
        scaled_likelihoods.columns = pd.Index(
            scaled_likelihoods.columns, name="label", dtype=data.label.dtype
        )
        return cast(pd.DataFrame, scaled_likelihoods)

    def _evidence_lower_bound(
        self,
        data: pd.DataFrame,
        probas: pd.DataFrame,
        priors: "pd.Series[Any]",
        errors: pd.DataFrame,
    ) -> float:
        # calculate joint probability log-likelihood expectation over probas
        joined = data.join(np.log(errors), on=["worker", "label"])  # type: ignore

        # escape boolean index/column names to prevent confusion between indexing by boolean array and iterable of names
        joined = joined.rename(columns={True: "True", False: "False"}, copy=False)
        priors = priors.rename(index={True: "True", False: "False"}, copy=False)

        joined.loc[:, priors.index] = joined.loc[:, priors.index].add(np.log(priors))  # type: ignore

        joined.set_index(["task", "worker"], inplace=True)
        joint_expectation = (
            (probas.rename(columns={True: "True", False: "False"}) * joined).sum().sum()
        )

        entropy = -(np.log(probas) * probas).sum().sum()
        return float(joint_expectation + entropy)

    def fit(self, data: pd.DataFrame) -> "DawidSkene":
        """Fits the model to the training data with the EM algorithm.
        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
        Returns:
            DawidSkene: self.
        """

        data = data[["task", "worker", "label"]]

        # Early exit
        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Initialization
        probas = MajorityVote().fit_predict_proba(data)
        priors = probas.mean()
        errors = self._m_step(data, probas)
        loss = -np.inf
        self.loss_history_ = []

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            errors = self._m_step(data, probas)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(
                data
            )
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        probas.columns = pd.Index(
            probas.columns, name="label", dtype=probas.columns.dtype
        )
        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self

    def fit_predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits the model to the training data and returns probability distributions of labels for each task.
        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
        Returns:
            DataFrame: Probability distributions of task labels.
                The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that the `task` true label is equal to `label`.
                Each probability is in the range from 0 to 1, all task probabilities must sum up to 1.
        """

        self.fit(data)
        assert self.probas_ is not None, "no probas_"
        return self.probas_

    def fit_predict(self, data: pd.DataFrame) -> "pd.Series[Any]":
        """Fits the model to the training data and returns the aggregated results.
        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
        Returns:
            Series: Task labels. The `pandas.Series` data is indexed by `task` so that `labels.loc[task]` is the most likely true label of tasks.
        """

        self.fit(data)
        assert self.labels_ is not None, "no labels_"
        return self.labels_


@attr.s
class OneCoinDawidSkene(DawidSkene):
    r"""The **one-coin Dawid-Skene** aggregation model works exactly the same as the original Dawid-Skene model
    based on the EM algorithm, except for calculating the workers' errors
    at the M-step of the algorithm.

    For the one-coin model, a worker confusion (error) matrix is parameterized by a single parameter $s_w$:
    $$
    e^w_{j,z_j}  = \begin{cases}
        s_{w} & y^w_j = z_j \\
        \frac{1 - s_{w}}{K - 1} & y^w_j \neq z_j
    \end{cases}
    $$
    where $e^w$ is a worker confusion (error) matrix of size $K \times K$ in case of the $K$ class classification,
    $z_j$ be a true task label, $y^w_j$ is a worker
    response to the task $j$, and $s_w$ is a worker skill (accuracy).

    In other words, the worker $w$ uses a single coin flip to decide their assignment. No matter what the true label is,
    the worker has the $s_w$ probability to assign the correct label, and
    has the $1 − s_w$ probability to randomly assign an incorrect label. For the one-coin model, it
    suffices to estimate $s_w$ for every worker $w$ and estimate $y^w_j$ for every task $j$. Because of its
    simplicity, the one-coin model is easier to estimate and enjoys better convergence properties.

    Parameters $p$, $e^w$, and latent variables $z$ are optimized with the Expectation-Maximization algorithm:
    1. **E-step**. Estimates the true task label probabilities using the specified workers' responses,
    the prior label probabilities, and the workers' error probability matrix.
    2. **M-step**. Calculates a worker skill as their accuracy according to the label probability.
    Then estimates the workers' error probability matrix by assigning user skills to error matrix row by row.

    Y. Zhang, X. Chen, D. Zhou, and M. I. Jordan. Spectral methods meet EM: A provably optimal algorithm for crowdsourcing.
    *Journal of Machine Learning Research. Vol. 17*, (2016), 1-44.

    <https://doi.org/10.48550/arXiv.1406.3824>

    Examples:
        >>> from crowdkit.aggregation import OneCoinDawidSkene
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> hds = OneCoinDawidSkene(100)
        >>> result = hds.fit_predict(df)
    """

    @staticmethod
    def _assign_skills(row: "pd.Series[Any]", skills: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns user skills to error matrix row by row.
        """
        num_categories = len(row)
        for column_name, _ in row.items():
            if column_name == row.name[1]:  # type: ignore
                row[column_name] = skills[row.name[0]]  # type: ignore
            else:
                row[column_name] = (1 - skills[row.name[0]]) / (num_categories - 1)  # type: ignore
        return row  # type: ignore

    @staticmethod
    def _process_skills_to_errors(
        data: pd.DataFrame, probas: pd.DataFrame, skills: "pd.Series[Any]"
    ) -> pd.DataFrame:
        errors = DawidSkene._m_step(data, probas)

        errors = errors.apply(OneCoinDawidSkene._assign_skills, args=(skills,), axis=1)  # type: ignore
        errors.clip(lower=_EPS, upper=1 - _EPS, inplace=True)

        return errors

    @staticmethod
    def _m_step(data: pd.DataFrame, probas: pd.DataFrame) -> "pd.Series[Any]":  # type: ignore
        """Performs M-step of Homogeneous Dawid-Skene algorithm.

        Calculates a worker skill as their accuracy according to the label probability.
        """
        skilled_data = data.copy()
        idx_cols, cols = pd.factorize(data["label"])
        idx_rows, rows = pd.factorize(data["task"])
        skilled_data["skill"] = (
            probas.reindex(rows, axis=0)
            .reindex(cols, axis=1)
            .to_numpy()[idx_rows, idx_cols]
        )
        skills = skilled_data.groupby(["worker"], sort=False)["skill"].mean()
        return skills

    def fit(self, data: pd.DataFrame) -> "OneCoinDawidSkene":
        """Fits the model to the training data with the EM algorithm.
        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
        Returns:
            DawidSkene: self.
        """

        data = data[["task", "worker", "label"]]

        # Early exit
        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Initialization
        probas = MajorityVote().fit_predict_proba(data)
        priors = probas.mean()
        skills = self._m_step(data, probas)
        errors = self._process_skills_to_errors(data, probas, skills)
        loss = -np.inf
        self.loss_history_ = []

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            skills = self._m_step(data, probas)
            errors = self._process_skills_to_errors(data, probas, skills)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(
                data
            )
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.skills_ = skills
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self
