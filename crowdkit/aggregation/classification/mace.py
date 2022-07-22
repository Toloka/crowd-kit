__all__ = ['MACE']

from typing import Any, List, Optional, Tuple
import attr
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.stats as sps
from scipy.special import digamma
from tqdm.auto import trange

from ..base import BaseClassificationAggregator


def normalize(x: NDArray[np.float64], smoothing: float) -> NDArray[np.float64]:
    """Normalize the rows of the matrix using the smoothing parameter.

    Args:
        x (np.ndarray): array to normalize
        smoothing (float): smoothing parameter

    Returns:
        np.ndarray: normalized array
    """
    norm = (x + smoothing).sum(axis=1)  # type: ignore
    return np.divide(  # type: ignore
        x + smoothing,
        norm[:, np.newaxis],
        out=np.zeros_like(x),
        where=~np.isclose(norm[:, np.newaxis], np.zeros_like(norm[:, np.newaxis])),
    )


def variational_normalize(x: NDArray[np.float64], hparams: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize the rows of the matrix using the MACE priors.

    Args:
        x (np.ndarray): array to normalize
        hparams (np.ndarray): prior parameters

    Returns:
        np.ndarray: normalized array
    """
    norm = (x + hparams).sum(axis=1)  # type: ignore
    norm = np.exp(digamma(norm))
    return np.divide(  # type: ignore
        np.exp(digamma(x + hparams)),
        norm[:, np.newaxis],
        out=np.zeros_like(x),
        where=~np.isclose(norm[:, np.newaxis], np.zeros_like(norm[:, np.newaxis])),
    )


def decode_distribution(gold_label_marginals: pd.DataFrame) -> pd.DataFrame:
    """Decode the distribution from marginals.

    Args:
        gold_label_marginals (pd.DataFrame): gold label marginals

    Returns:
        pd.DataFrame: decoded distribution
    """

    return gold_label_marginals.div(gold_label_marginals.sum(axis=1), axis=0)


@attr.s
class MACE(BaseClassificationAggregator):
    r"""Multi-Annotator Competence Estimation.

    Probabilistic model that associates each worker with a probability distribution over the labels.
    For each task, a worker might be in a spamming or not spamming state. If the worker is not
    spamming, they yield a correct label. If the worker is spamming, they answer according
    to their probability distribution.

    We assume that the correct label $T_i$ comes from a discrete uniform distribution. When a worker
    annotates the task, they are in the spamming state with probability
    $\operatorname{Bernoulli}(1 - \theta_w)$. So, if their state $s_w = 0$, their response
    $A_{iw} = T_i$. Otherwise, their response $A_{iw}$ is drawn from a multinomial
    distribution with parameters $\xi_w$.

    ![MACE latent label model](https://tlk.s3.yandex.net/crowd-kit/docs/mace_llm.png)

    The model can be enhanced by adding a Beta prior over $\theta_w$ and Diriclet
    prior over $\xi_w$.

    D. Hovy, T. Berg-Kirkpatrick, A. Vaswani and E. Hovy. Learning Whom to Trust with MACE.
    In *Proceedings of NAACL-HLT*, Atlanta, GA, USA (2013), 1120–1130.

    <https://aclanthology.org/N13-1132.pdf>

    Args:
        n_restarts (int): The of algorithms optimization runs. The final parameters are ones
            that gave the best log likelihood. When a single run takes too long, it is fine
            to set this parameter to 1. Default: 10.
        n_iter (int): The number of EM iterations for each optimization run. Default: 50.
        method (str): The method to use for the M-step. Either 'vb' or 'em'. 'vb' means
            optimization through variational Bayes using priors. 'em' stands for
            straightforward Expectation-Maximization. Default: 'vb'.
        smoothing (float): The smoothing parameter for the normalization. Default: 0.1.
        alpha (float): The prior parameter for the Beta distribution over $\theta_w$. Default: 0.5.
        beta (float): The prior parameter for the Beta distribution over $\theta_w$. Default: 0.5.
        default_noise (float): The default noise parameter for the initialization. Default: 0.5.
        verbose (bool): Whether to print progress. 0 — no progress bar, 1 — only for restarts,
         2 — for both restarts and optimization. Default: 0.

    Examples:
        >>> from crowdkit.aggregation import MACE
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> mace = MACE()
        >>> result = mace.fit_predict(df)

    Attributes:
        labels_ (Optional[pd.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        probas_ (Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1

        spamming_ (Optional[pd.Series]): Posterior distribution of workers' spamming states.

        thetas_ (Optional[pandas.core.frame.DataFrame]): Posterior distribution of workers'
            spamming labels.
    """

    n_restarts: int = attr.ib(default=10)
    n_iter: int = attr.ib(default=50)
    method: str = attr.ib(default="vb")
    default_noise: float = attr.ib(default=0.5)
    alpha: float = attr.ib(default=0.5)
    beta: float = attr.ib(default=0.5)
    random_state: int = attr.ib(default=0)
    verbose: int = attr.ib(default=0)

    spamming_: NDArray[np.float64] = attr.ib(init=False)
    thetas_: NDArray[np.float64] = attr.ib(init=False)
    theta_priors_: Optional[NDArray[np.float64]] = attr.ib(init=False)
    strategy_priors_: Optional[NDArray[np.float64]] = attr.ib(init=False)
    smoothing_: float = attr.ib(init=False)
    probas_: Optional[pd.DataFrame] = attr.ib(init=False)

    def fit(self, data: pd.DataFrame) -> 'MACE':
        """Fits the MACE model.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            MACE: The fitted MACE model.
        """

        workers, worker_names = pd.factorize(data['worker'])
        labels, label_names = pd.factorize(data['label'])
        tasks, task_names = pd.factorize(data['task'])

        n_workers = len(worker_names)
        n_labels = len(label_names)

        self.smoothing_ = 0.01 / n_labels

        annotation = data.copy(deep=True)

        best_log_marginal_likelihood = -np.inf
        restarts_progress = (
            trange(self.n_restarts) if self.verbose > 0 else range(self.n_restarts)
        )
        if self.verbose > 0:
            restarts_progress.set_description('Restarts')
        for _ in restarts_progress:
            self._initialize(n_workers, n_labels)
            (
                log_marginal_likelihood,
                gold_label_marginals,
                strategy_expected_counts,
                knowing_expected_counts,
            ) = self._e_step(
                annotation,
                task_names,
                worker_names,
                label_names,
                tasks,
                workers,
                labels,
            )
            iter_progress = (
                trange(self.n_iter) if self.verbose > 1 else range(self.n_iter)
            )
            for _ in iter_progress:
                if self.method == 'vb':
                    self._variational_m_step(
                        knowing_expected_counts, strategy_expected_counts
                    )
                else:
                    self._m_step(knowing_expected_counts, strategy_expected_counts)
                (
                    log_marginal_likelihood,
                    gold_label_marginals,
                    strategy_expected_counts,
                    knowing_expected_counts,
                ) = self._e_step(
                    annotation,
                    task_names,
                    worker_names,
                    label_names,
                    tasks,
                    workers,
                    labels,
                )
                if self.verbose > 1:
                    iter_progress.set_postfix(
                        {'log_marginal_likelihood': round(log_marginal_likelihood, 5)}
                    )
            if log_marginal_likelihood > best_log_marginal_likelihood:
                best_log_marginal_likelihood = log_marginal_likelihood
                best_thetas = self.thetas_.copy()
                best_spamming = self.spamming_.copy()

        self.thetas_ = best_thetas
        self.spamming_ = best_spamming
        _, gold_label_marginals, _, _ = self._e_step(
            annotation, task_names, worker_names, label_names, tasks, workers, labels
        )

        self.probas_ = decode_distribution(gold_label_marginals)
        self.labels_ = self.probas_.idxmax(axis='columns')
        self.labels_.index.name = 'task'

        return self

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Fits the MACE model and returns the labels.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            pandas.Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        return self.fit(data).labels_

    def fit_predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the MACE model and returns the label probability distributions.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            pandas.DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        return self.fit(data).probas_

    def _initialize(self, n_workers: int, n_labels: int) -> None:
        """Initialize the MACE parameters.

        Args:
            n_workers (int): The number of workers.
            n_labels (int): The number of labels.

        Returns:
            None
        """

        self.spamming_ = sps.uniform(1, 1 + self.default_noise).rvs(
            size=(n_workers, 2),
            random_state=self.random_state,
        )
        self.thetas_ = sps.uniform(1, 1 + self.default_noise).rvs(
            size=(n_workers, n_labels),
            random_state=self.random_state
        )

        self.spamming_ = self.spamming_ / self.spamming_.sum(axis=1, keepdims=True)
        self.thetas_ = self.thetas_ / self.thetas_.sum(axis=1, keepdims=True)

        if self.method == 'vb':
            self.theta_priors_ = np.empty((n_workers, 2))
            self.theta_priors_[:, 0] = self.alpha
            self.theta_priors_[:, 1] = self.beta

            self.strategy_priors_ = np.ones((n_workers, n_labels)) * 10.0

    def _e_step(
        self,
        annotation: pd.DataFrame,
        task_names: List[Any],
        worker_names: List[Any],
        label_names: List[Any],
        tasks: NDArray[np.int64],
        workers: NDArray[np.int64],
        labels: NDArray[np.int64],
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """E-step of the MACE algorithm.

        Args:
            annotation (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            task_names (List[Any]): The task names.
            worker_names (List[Any]): The worker names.
            label_names (List[Any]): The label names.
            tasks (np.ndarray): The tasks ids in the annotation.
            workers (np.ndarray): The workers ids in the annotation.
            labels (np.ndarray): The labels ids in the annotation.

        Returns:
            Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The log marginal likelihood,
                gold label marginals,strategy expected counts and knowing expected counts.
        """
        gold_label_marginals = pd.DataFrame(
            np.zeros((len(task_names), len(label_names))),
            index=task_names,
            columns=label_names,
        )

        knowing_expected_counts = pd.DataFrame(
            np.zeros((len(worker_names), 2)),
            index=worker_names,
            columns=['knowing_expected_count_0', 'knowing_expected_count_1'],
        )

        for label_idx, label in enumerate(label_names):
            annotation['gold_marginal'] = self.spamming_[workers, 0] * self.thetas_[
                workers, labels
            ] + self.spamming_[workers, 1] * (label_idx == labels)
            gold_label_marginals[label] = annotation.groupby('task').prod()[
                'gold_marginal'
            ] / len(label_names)

        instance_marginals = gold_label_marginals.sum(axis=1)
        log_marginal_likelihood = np.log(instance_marginals + 1e-8).sum()

        annotation['strategy_marginal'] = 0.0
        for label in range(len(label_names)):
            annotation['strategy_marginal'] += gold_label_marginals.values[
                tasks, label
            ] / (
                self.spamming_[workers, 0] * self.thetas_[workers, labels]
                + self.spamming_[workers, 1] * (labels == label)
            )

        annotation['strategy_marginal'] = (
            annotation['strategy_marginal']
            * self.spamming_[workers, 0]
            * self.thetas_[workers, labels]
        )

        annotation.set_index('task', inplace=True)
        annotation['instance_marginal'] = instance_marginals
        annotation.reset_index(inplace=True)

        annotation['strategy_marginal'] = (
            annotation['strategy_marginal'] / annotation['instance_marginal']
        )

        strategy_expected_counts = (
            annotation.groupby(['worker', 'label']).sum()['strategy_marginal'].unstack().fillna(0.0)
        )

        knowing_expected_counts['knowing_expected_count_0'] = annotation.groupby(
            'worker'
        ).sum()['strategy_marginal']

        annotation['knowing_expected_counts'] = (
            gold_label_marginals.values[tasks, labels].ravel()
            * self.spamming_[workers, 1]
            / (
                self.spamming_[workers, 0] * self.thetas_[workers, labels]
                + self.spamming_[workers, 1]
            )
        ) / instance_marginals.values[tasks]
        knowing_expected_counts['knowing_expected_count_1'] = annotation.groupby(
            'worker'
        ).sum()['knowing_expected_counts']

        return (
            log_marginal_likelihood,
            gold_label_marginals,
            strategy_expected_counts,
            knowing_expected_counts,
        )

    def _m_step(
        self,
        knowing_expected_counts: pd.DataFrame,
        strategy_expected_counts: pd.DataFrame,
    ) -> None:
        """
        M-step of the MACE algorithm.

        Args:
            knowing_expected_counts (DataFrame): The knowing expected counts.
            strategy_expected_counts (DataFrame): The strategy expected counts.

        Returns:
            None
        """
        self.spamming_ = normalize(knowing_expected_counts.values, self.smoothing_)
        self.thetas_ = normalize(strategy_expected_counts.values, self.smoothing_)

    def _variational_m_step(
        self,
        knowing_expected_counts: pd.DataFrame,
        strategy_expected_counts: pd.DataFrame,
    ) -> None:
        """
        Variational M-step of the MACE algorithm.

        Args:
            knowing_expected_counts (DataFrame): The knowing expected counts.
            strategy_expected_counts (DataFrame): The strategy expected counts.

        Returns:
            None
        """
        assert self.theta_priors_ is not None
        self.spamming_ = variational_normalize(
            knowing_expected_counts.values, self.theta_priors_
        )
        assert self.strategy_priors_ is not None
        self.thetas_ = variational_normalize(
            strategy_expected_counts.values, self.strategy_priors_
        )
