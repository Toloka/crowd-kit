import attr
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
import scipy.stats as sps
from typing import Optional, Tuple

from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote


@attr.attrs(auto_attribs=True)
class MMSR(BaseAggregator):
    """
    Matrix Mean-Subsequence-Reduced Algorithm
    Qianqian Ma and Alex Olshevsky. 2020.
    Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion
    34th Conference on Neural Information Processing Systems (NeurIPS 2020)
    https://arxiv.org/abs/2010.12181

    Input:
    - crowd-dataframe [task, performer, label]
    - n_iter - optional, the number of iterations to stop after
    - eps - optional, threshold in change to stop the algorithm
    Output:
    - result-dataframe - [task, label]
    """
    n_iter: int = 10000
    eps: float = 1e-10
    random_state: Optional[int] = 0
    _observation_matrix: np.ndarray = np.array([])
    _covariation_matrix: np.ndarray = np.array([])
    _n_common_tasks: np.ndarray = np.array([])
    _n_performers: int = 0
    _n_tasks: int = 0
    _n_labels: int = 0
    _labels_mapping: dict = dict()
    _performers_mapping: dict = dict()
    _tasks_mapping: dict = dict()

    def fit(self, answers: pd.DataFrame) -> 'MMSR':
        """Calculates the skill for each performers through rank-one matrix completion
        The calculated skills are stored in an instance of the class and can be obtained by the field 'performers_skills'
        After 'fit' you can get 'performer_skills' from class field.

        Args:
            answers(pandas.DataFrame): Frame contains performers answers. One row per answer.
                Should contain columns 'performer', 'task', 'label'.
        Returns:
            MMSR: self for call next methods

        Raises:
            TypeError: If the input datasets are not of type pandas.DataFrame.
            AssertionError: If there is some collumn missing in 'dataframes'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        return self

    def predict(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Predict correct labels for tasks. Using calculated performers skill, stored in self instance.
        After 'predict' you can get probabilities for all labels from class field 'probas'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Predicted label for each task.
                - task - unique values from input dataset
                - label - most likely label

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers' in 'fit'.
        """
        self._answers_base_checks(answers)
        return self._predict_impl(answers)

    def predict_proba(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Calculates probabilities for each label of task.
        After 'predict_proba' you can get predicted labels from class field 'tasks_labels'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task and the likelihood of correctness.
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers' in 'fit'.
        """
        self._answers_base_checks(answers)
        self._predict_impl(answers)
        return self.probas

    def fit_predict(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Performes 'fit' and 'predict' in one call.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Predicted label for each task.
                - task - unique values from input dataset
                - label - most likely label

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        return self._predict_impl(answers)

    def fit_predict_proba(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Performes 'fit' and 'predict_proba' in one call.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task and the likelihood of correctness.
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        self._predict_impl(answers)
        return self.probas

    def _fit_impl(self, answers: pd.DataFrame) -> None:
        self._construnct_covariation_matrix(answers)
        self._m_msr()

    def _predict_impl(self, answers: pd.DataFrame) -> pd.DataFrame:
        weighted_mv = MajorityVote()
        labels = weighted_mv.fit_predict(answers, self._performers_weights)
        self.tasks_labels = labels
        self.probas = weighted_mv.probas
        return self.tasks_labels

    def _m_msr(self) -> None:
        F_param = int(np.floor(self._sparsity / 2)) - 1
        n, m = self._covariation_matrix.shape
        u = sps.uniform.rvs(size=(n, 1), random_state=self.random_state)
        v = sps.uniform.rvs(size=(m, 1), random_state=self.random_state)
        observed_entries = np.abs(np.sign(self._n_common_tasks)) == 1
        X = np.abs(self._covariation_matrix)

        for _ in range(self.n_iter):
            v_prev = v
            u_prev = u
            for j in range(n):
                target_v = X[:, j]
                target_v = target_v[observed_entries[:, j]] / u[observed_entries[:, j]]

                y = self._remove_largest_and_smallest_F_value(target_v, F_param, v[j][0], self._n_tasks)
                if len(y) == 0:
                    v[j] = v[j]
                else:
                    v[j][0] = y.mean()

            for i in range(m):
                target_u = X[i, :].reshape(-1, 1)
                target_u = target_u[observed_entries[i, :].ravel()] / v[observed_entries[i, :].ravel()]
                y = self._remove_largest_and_smallest_F_value(target_u, F_param, u[i][0], self._n_tasks)
                if len(y) == 0:
                    u[i] = u[i]
                else:
                    u[i][0] = y.mean()

            if np.linalg.norm(u @ v.T - u_prev @ v_prev.T, ord='fro') < self.eps:
                break

        k = np.sqrt(np.linalg.norm(u) / np.linalg.norm(v))
        x_track_1 = u / k
        x_track_2 = self._sign_determination_valid(self._covariation_matrix, x_track_1)
        x_track_3 = np.minimum(x_track_2, 1 - 1. / np.sqrt(self._n_tasks))
        x_MSR = np.maximum(x_track_3, -1 / (self._n_labels - 1) + 1. / np.sqrt(self._n_tasks))

        performers_probas = x_MSR * (self._n_labels - 1) / (self._n_labels) + 1 / self._n_labels
        performers_probas = performers_probas.ravel()
        self._set_skills_from_array(performers_probas)
        self._set_performers_weights()

    def _set_skills_from_array(self, array: np.ndarray) -> None:
        inverse_performers_mapping = {ind: performer for performer, ind in self._performers_mapping.items()}
        self.performers_skills = pd.DataFrame(
            [
                [inverse_performers_mapping[ind], array[ind]]
                for ind in range(len(array))
            ],
            columns=['performer', 'skill']
        )

    def _set_performers_weights(self) -> None:
        self._performers_weights = self.performers_skills.copy().rename(columns={'skill': 'weight'})
        self._performers_weights['weight'] = self._performers_weights['weight'] * (self._n_labels - 1) / (self._n_labels) + 1 / self._n_labels

    @staticmethod
    def _sign_determination_valid(C: np.ndarray, s_abs: np.ndarray) -> np.ndarray:
        S = np.sign(C)
        n = len(s_abs)

        valid_idx = np.where(np.sum(C, axis=1) != 0)[0]
        S_valid = S[valid_idx[:, None], valid_idx]
        k = S_valid.shape[0]
        upper_idx = np.triu(np.ones(shape=(k, k)))
        S_valid_upper = S_valid * upper_idx
        new_node_end_I, new_node_end_J = np.where(S_valid_upper == 1)
        S_valid[S_valid == 1] = 0
        I = np.eye(k)
        S_valid_new = I[new_node_end_I, :] + I[new_node_end_J, :]
        m = S_valid_new.shape[0]
        A = np.vstack((np.hstack((np.abs(S_valid), S_valid_new.T)), np.hstack((S_valid_new, np.zeros(shape=(m, m))))))
        n_new = A.shape[0]
        W = (1. / np.sum(A, axis=1)).reshape(-1, 1) @ np.ones(shape=(1, n_new)) * A
        D, V = sla.eigs(W + np.eye(n_new), 1, which='SM')
        V = V.real
        sign_vector = np.sign(V)
        s_sign = np.zeros(shape=(n, 1))
        s_sign[valid_idx] = np.sign(np.sum(sign_vector[:k])) * s_abs[valid_idx] * sign_vector[:k]
        return s_sign

    @staticmethod
    def _remove_largest_and_smallest_F_value(x, F, a, n_tasks) -> np.ndarray:
        y = np.sort(x, axis=0)
        if np.sum(y < a) < F:
            y = y[y[:, 0] >= a]
        else:
            y = y[F:]

        m = y.shape[0]
        if np.sum(y > a) < F:
            y = y[y[:, 0] <= a]
        else:
            y = np.concatenate((y[:m - F], y[m:]), axis=0)
        if len(y) == 1 and y[0][0] == 0:
            y[0][0] = 1 / np.sqrt(n_tasks)
        return y

    def _construnct_covariation_matrix(self, answers: pd.DataFrame) -> Tuple[np.ndarray]:
        labels = pd.unique(answers.label)
        self._n_labels = len(labels)
        self._labels_mapping = {labels[idx]: idx + 1 for idx in range(self._n_labels)}

        performers = pd.unique(answers.performer)
        self._n_performers = len(performers)
        self._performers_mapping = {performers[idx]: idx for idx in range(self._n_performers)}

        tasks = pd.unique(answers.task)
        self._n_tasks = len(tasks)
        self._tasks_mapping = {tasks[idx]: idx for idx in range(self._n_tasks)}

        self._observation_matrix = np.zeros(shape=(self._n_performers, self._n_tasks))
        for i, row in answers.iterrows():
            self._observation_matrix[self._performers_mapping[row['performer']]][self._tasks_mapping[row['task']]] = self._labels_mapping[row['label']]

        self._n_common_tasks = np.sign(self._observation_matrix) @ np.sign(self._observation_matrix).T
        self._n_common_tasks -= np.diag(np.diag(self._n_common_tasks))
        self._sparsity = np.min(np.sign(self._n_common_tasks).sum(axis=0))

        # Can we rewrite it in matrix operations?
        self._covariation_matrix = np.zeros(shape=(self._n_performers, self._n_performers))
        for i in range(self._n_performers):
            for j in range(self._n_performers):
                if self._n_common_tasks[i][j]:
                    valid_idx = np.sign(self._observation_matrix[i]) * np.sign(self._observation_matrix[j])
                    self._covariation_matrix[i][j] = np.sum((self._observation_matrix[i] == self._observation_matrix[j]) * valid_idx) / self._n_common_tasks[i][j]

        self._covariation_matrix *= self._n_labels / (self._n_labels - 1)
        self._covariation_matrix -= np.ones(shape=(self._n_performers, self._n_performers)) / (self._n_labels - 1)
