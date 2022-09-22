__all__ = [
    'DawidSkene',
    'OneCoinDawidSkene'
]

from typing import List, Optional

import attr
import numpy as np
import pandas as pd

from .majority_vote import MajorityVote
from ..base import BaseClassificationAggregator
from ..utils import get_most_probable_labels, named_series_attrib

_EPS = np.float_power(10, -10)


@attr.s
class DawidSkene(BaseClassificationAggregator):
    r"""Dawid-Skene aggregation model.

    Probabilistic model that parametrizes workers' level of expertise through confusion matrices.

    Let $e^w$ be a worker's confusion (error) matrix of size $K \times K$ in case of $K$ class classification,
    $p$ be a vector of prior classes probabilities, $z_j$ be a true task's label, and $y^w_j$ be a worker's
    answer for the task $j$. The relationships between these parameters are represented by the following latent
    label model.

    ![Dawid-Skene latent label model](https://tlk.s3.yandex.net/crowd-kit/docs/ds_llm.png)

    Here the prior true label probability is
    $$
    \operatorname{Pr}(z_j = c) = p[c],
    $$
    and the distribution on the worker's responses given the true label $c$ is represented by the
    corresponding column of the error matrix:
    $$
    \operatorname{Pr}(y_j^w = k | z_j = c) = e^w[k, c].
    $$

    Parameters $p$ and $e^w$ and latent variables $z$ are optimized through the Expectation-Maximization algorithm.

    A. Philip Dawid and Allan M. Skene. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    *Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28*, 1 (1979), 20–28.

    <https://doi.org/10.2307/2346806>

    Args:
        n_iter: The number of EM iterations.

    Examples:
        >>> from crowdkit.aggregation import DawidSkene
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> ds = DawidSkene(100)
        >>> result = ds.fit_predict(df)

    Attributes:
        labels_ (Optional[pd.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        probas_ (Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1

        priors_ (Optional[pd.Series]): A prior label distribution.
            A pandas.Series indexed by labels and holding corresponding label's
            probability of occurrence. Each probability is between 0 and 1,
            all probabilities should sum up to 1

        errors_ (Optional[pandas.core.frame.DataFrame]): Workers' error matrices.
            A pandas.DataFrame indexed by `worker` and `label` with a column for every
            label_id found in `data` such that `result.loc[worker, observed_label, true_label]`
            is the probability of `worker` producing an `observed_label` given that a task's
            true label is `true_label`
    """

    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-5)

    probas_: Optional[pd.DataFrame] = attr.ib(init=False)
    priors_: Optional[pd.Series] = named_series_attrib(name='prior')
    # labels_
    errors_: Optional[pd.DataFrame] = attr.ib(init=False)
    loss_history_: List[float] = attr.ib(init=False)

    @staticmethod
    def _m_step(data: pd.DataFrame, probas: pd.DataFrame) -> pd.DataFrame:
        """Perform M-step of Dawid-Skene algorithm.

        Given workers' answers and tasks' true labels probabilities estimates
        worker's errors probabilities matrix.
        """
        joined = data.join(probas, on='task')
        joined.drop(columns=['task'], inplace=True)

        errors = joined.groupby(['worker', 'label'], sort=False).sum()
        errors.clip(lower=_EPS, inplace=True)
        errors /= errors.groupby('worker', sort=False).sum()

        return errors

    @staticmethod
    def _e_step(data: pd.DataFrame, priors: pd.Series, errors: pd.DataFrame) -> pd.DataFrame:
        """
        Perform E-step of Dawid-Skene algorithm.

        Given worker's answers, labels' prior probabilities and worker's worker's
        errors probabilities matrix estimates tasks' true labels probabilities.
        """

        # We have to multiply lots of probabilities and such products are known to converge
        # to zero exponentially fast. To avoid floating-point precision problems we work with
        # logs of original values
        joined = data.join(np.log2(errors), on=['worker', 'label'])
        joined.drop(columns=['worker', 'label'], inplace=True)
        log_likelihoods = np.log2(priors) + joined.groupby('task', sort=False).sum()

        # Exponentiating log_likelihoods 'as is' may still get us beyond our precision.
        # So we shift every row of log_likelihoods by a constant (which is equivalent to
        # multiplying likelihoods rows by a constant) so that max log_likelihood in each
        # row is equal to 0. This trick ensures proper scaling after exponentiating and
        # does not affect the result of E-step
        scaled_likelihoods = np.exp2(log_likelihoods.sub(log_likelihoods.max(axis=1), axis=0))
        return scaled_likelihoods.div(scaled_likelihoods.sum(axis=1), axis=0)

    def _evidence_lower_bound(self, data: pd.DataFrame, probas: pd.DataFrame, priors: pd.Series, errors: pd.DataFrame) -> float:
        # calculate joint probability log-likelihood expectation over probas
        joined = data.join(np.log(errors), on=['worker', 'label'])

        # escape boolean index/column names to prevent confusion between indexing by boolean array and iterable of names
        joined = joined.rename(columns={True: 'True', False: 'False'}, copy=False)
        priors = priors.rename(index={True: 'True', False: 'False'}, copy=False)

        joined.loc[:, priors.index] = joined.loc[:, priors.index].add(np.log(priors))

        joined.set_index(['task', 'worker'], inplace=True)
        joint_expectation = (probas.rename(columns={True: 'True', False: 'False'}) * joined).sum().sum()

        entropy = -(np.log(probas) * probas).sum().sum()
        return float(joint_expectation + entropy)

    def fit(self, data: pd.DataFrame) -> 'DawidSkene':
        """Fit the model through the EM-algorithm.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DawidSkene: self.
        """

        data = data[['task', 'worker', 'label']]

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
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        probas.columns = pd.Index(probas.columns, name='label')
        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self

    def fit_predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and return probability distributions on labels for each task.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

        return self.fit(data).probas_

    def fit_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and return aggregated results.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """

        return self.fit(data).labels_


@attr.s
class OneCoinDawidSkene(DawidSkene):
    r"""One-coin Dawid-Skene aggregation model.

    This model works exactly like original Dawid-Skene model based on EM Algorithm except for workers' error calculation
    on M-step of the algorithm.

    First the workers' skills are calculated as their accuracy in accordance with labels probability.
    Let $e^w$ be a worker's confusion (error) matrix of size $K \times K$ in case of $K$ class classification,
    $p$ be a vector of prior classes probabilities, $z_j$ be a true task's label, and $y^w_j$ be a worker's
    answer for the task $j$. Let s_{w} be a worker's skill (accuracy). Then the error
    $$
    e^w_{j,z_j}  = \begin{cases}
        s_{w} & y^w_j = z_j \\
        \frac{1 - s_{w}}{K - 1} & y^w_j \neq z_j
    \end{cases}
    $$

    Args:
        n_iter: The number of EM iterations.

    Examples:
        >>> from crowdkit.aggregation import OneCoinDawidSkene
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> hds = OneCoinDawidSkene(100)
        >>> result = hds.fit_predict(df)
    Attributes:
        labels_ (Optional[pd.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        probas_ (Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1

        priors_ (Optional[pd.Series]): A prior label distribution.
            A pandas.Series indexed by labels and holding corresponding label's
            probability of occurrence. Each probability is between 0 and 1,
            all probabilities should sum up to 1

        errors_ (Optional[pandas.core.frame.DataFrame]): Workers' error matrices.
            A pandas.DataFrame indexed by `worker` and `label` with a column for every
            label_id found in `data` such that `result.loc[worker, observed_label, true_label]`
            is the probability of `worker` producing an `observed_label` given that a task's
            true label is `true_label`

        skills_ (Optional[pd.Series]): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
    """

    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-5)

    probas_: pd.DataFrame = attr.ib(init=False)
    priors_: pd.Series = named_series_attrib(name='prior')
    errors_: pd.DataFrame = attr.ib(init=False)
    skills_: pd.Series = attr.ib(init=False)
    loss_history_: List[float] = attr.ib(init=False)

    @staticmethod
    def _assign_skills(row: pd.Series, skills: pd.DataFrame) -> pd.DataFrame:
        """
        Assign user skills to error matrix row by row.
        """
        num_categories = len(row)
        for column_name, _ in row.items():
            if column_name == row.name[1]:
                row[column_name] = skills[row.name[0]]
            else:
                row[column_name] = (1 - skills[row.name[0]]) / (num_categories - 1)
        return row

    @staticmethod
    def _process_skills_to_errors(data: pd.DataFrame, probas: pd.DataFrame, skills: pd.Series) -> pd.DataFrame:
        errors = DawidSkene._m_step(data, probas)

        errors = errors.apply(OneCoinDawidSkene._assign_skills, args=(skills,), axis=1)
        errors.clip(lower=_EPS, upper=1 - _EPS, inplace=True)

        return errors

    @staticmethod
    def _m_step(data: pd.DataFrame, probas: pd.DataFrame) -> pd.Series:
        """Perform M-step of Homogeneous Dawid-Skene algorithm.

        Given workers' answers and tasks' true labels probabilities estimates
        worker's errors probabilities matrix.
        """
        skilled_data = data.copy()
        idx_cols, cols = pd.factorize(data['label'])
        idx_rows, rows = pd.factorize(data['task'])
        skilled_data['skill'] = probas.reindex(rows, axis=0).reindex(cols, axis=1).to_numpy()[idx_rows, idx_cols]
        skills = skilled_data.groupby(['worker'], sort=False)['skill'].mean()
        return skills

    def fit(self, data: pd.DataFrame) -> 'OneCoinDawidSkene':
        """Fit the model through the EM-algorithm.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DawidSkene: self.
        """

        data = data[['task', 'worker', 'label']]

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
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
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
