__all__ = ['ZeroBasedSkill']

import attr
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from . import annotations
from .annotations import Annotation, manage_docstring
from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote
from .utils import get_accuracy


@attr.attrs(auto_attribs=True)
class ZeroBasedSkill(BaseAggregator):
    """The Zero-Based Skill aggregation model

    Performs weighted majority voting on tasks. After processing a pool of tasks,
    re-estimates performers' skills according to the correctness of their answers.
    Repeats this process until labels do not change or the number of iterations exceeds.

    It's necessary that all performers in a dataset that send to 'predict' existed in answers
    the dataset that was sent to 'fit'.
    """

    n_iter: int = 100
    lr_init: float = 1.0
    lr_steps_to_reduce: int = 20
    lr_reduce_factor: float = 0.5
    eps: float = 1e-5

    # Available after fit
    skills_: annotations.OPTIONAL_SKILLS = attr.ib(init=False)

    # Available after predict or predict_proba
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    labels_: annotations.OPTIONAL_LABELS = attr.ib(init=False)

    def _init_skills(self, data: annotations.LABELED_DATA) -> annotations.SKILLS:
        skill_value = 1 / data.label.unique().size + self.eps
        skill_index = pd.Index(data.performer.unique(), name='performer')
        return pd.Series(skill_value, index=skill_index)

    @manage_docstring
    def _apply(self, data: annotations.LABELED_DATA) -> Annotation(type='ZeroBasedSkill', title='self'):
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, self.skills_)
        self.labels_ = mv.labels_
        self.probas_ = mv.probas_
        return self

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> Annotation(type='ZeroBasedSkill', title='self'):

        # Initialization
        data = data[['task', 'performer', 'label']]
        skills = self._init_skills(data)
        mv = MajorityVote()

        # Updating skills and re-fitting majority vote n_iter times
        learning_rate = self.lr_init
        for iteration in range(1, self.n_iter + 1):
            if iteration % self.lr_steps_to_reduce == 0:
                learning_rate *= self.lr_reduce_factor
            mv.fit(data, skills=skills)
            skills = skills + learning_rate * (get_accuracy(data, mv.labels_, by='performer') - skills)

        # Saving results
        self.skills_ = skills

        return self

    @manage_docstring
    def predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self._apply(data).labels_

    @manage_docstring
    def predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self._apply(data).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self.fit(data).predict(data)

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data).predict_proba(data)
