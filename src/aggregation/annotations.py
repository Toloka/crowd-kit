"""
This module contains reusable annotations that encapsulate both typing
and description for commonly used parameters. These annotations are
used to automatically generate stub files with proper docstrings
"""

import inspect
import textwrap
from io import StringIO
from typing import Dict, Optional, Type

import attr
import pandas as pd


@attr.s
class Annotation:
    type: Optional[Type] = attr.ib(default=None)
    title: Optional[str] = attr.ib(default=None)
    description: Optional[str] = attr.ib(default=None)

    def format_google_style_attribute(self, name: str) -> str:
        type_str = f' ({getattr(self.type, "__name__", str(self.type))})' if self.type else ''
        title = f' {self.title}\n' if self.title else '\n'
        description_str = textwrap.indent(f'{self.description}\n', ' ' * 4).lstrip('\n') if self.description else ''
        return f'{name}{type_str}:{title}{description_str}'

    def format_google_style_return(self):
        type_str = f'{getattr(self.type, "__name__", str(self.type))}' if self.type else ''
        title = f' {self.title}\n' if self.title else '\n'
        description_str = textwrap.indent(f'{self.description}\n', ' ' * 4).lstrip('\n') if self.description else ''
        return f'{type_str}:{title}{description_str}'


def manage_docstring(obj):

    attributes: Dict[str, Annotation] = {}
    new_annotations = {}

    for key, value in getattr(obj, '__annotations__', {}).items():
        if isinstance(value, Annotation):
            attributes[key] = value
            if value.type is not None:
                new_annotations[key] = value.type
        else:
            new_annotations[key] = value

    return_section = attributes.pop('return', None)

    sio = StringIO()
    sio.write(inspect.cleandoc(obj.__doc__ or ''))

    if attributes:
        sio.write('\nArgs:\n' if inspect.isfunction(obj) else '\nAttributes:\n')
        for key, ann in attributes.items():
            sio.write(textwrap.indent(ann.format_google_style_attribute(key), ' ' * 4))

    if return_section:
        sio.write('Returns:\n')
        sio.write(textwrap.indent(return_section.format_google_style_return(), ' ' * 4))

    obj.__annotations__ = new_annotations
    obj.__doc__ = sio.getvalue()
    return obj


# Input data descriptions


EMBEDDED_DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' outputs with their embeddings",
    description='A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.'
)

LABELED_DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' labeling results",
    description='A pandas.DataFrame containing `task`, `performer` and `label` columns.',
)


PAIRWISE_DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' pairwise comparison results",
    description=textwrap.dedent('''
        A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns'.
        For each row `label` must be equal to either `left` or `right`.
    ''')
)


# Commonly used types

LABEL_PRIORS = Annotation(
    type=pd.Series,
    title='A prior label distribution',
    description=textwrap.dedent('''
        A pandas.Series indexed by labels and holding corresponding label's
        probability of occurrence. Each probability is between 0 and 1,
        all probabilities should sum up to 1
    '''),
)

LABEL_SCORES = Annotation(
    type=pd.Series,
    title="'Labels' scores",
    description="A pandas.Series index by labels and holding corresponding label's scores",
)

TASKS_EMBEDDINGS = Annotation(
    type=pd.DataFrame,
    title="Tasks' embeddings",
    description=textwrap.dedent("A pandas.DataFrame indexed by `task` with a single column `embedding`."),
)

TASKS_LABELS = Annotation(
    type=pd.DataFrame,
    title="Tasks' most likely true labels",
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` such that `labels.loc[task]`
        is the tasks's most likely true label.
    '''),
)

TASKS_LABEL_PROBAS = Annotation(
    type=pd.DataFrame,
    title="Tasks' true label probability distributions",
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
        is the probability of `task`'s true label to be equal to `label`. Each
        probability is between 0 and 1, all task's probabilities should sum up to 1
    '''),
)

TASKS_LABEL_SCORES = Annotation(
    type=pd.DataFrame,
    title="Tasks' true label scores",
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
        is the score of `label` for `task`.
    '''),
)

TASKS_TRUE_LABELS = Annotation(
    type=pd.Series,
    title="Tasks' ground truth labels",
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` such that `labels.loc[task]`
        is the tasks's ground truth label.
    '''),
)


SKILLS = Annotation(
    type=pd.Series,
    title="Performers' skills",
    description="A pandas.Series index by performers and holding corresponding performer's skill",
)


ERRORS = Annotation(
    type=pd.DataFrame,
    title="Performers' error matrices",
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `performer` and `label` with a column for every
        label_id found in `data` such that `result.loc[performer, observed_label, true_label]`
        is the probability of `performer` producing an `observed_label` given that a task's
        true label is `true_label`
    '''),
)


WEIGHTED_DATA = Annotation(
    type=pd.DataFrame,
    title='Input data',
    description='A pandas.DataFrame containing `task`, `performer`, `label` and optionally `weight` columns',
)

WEIGHTS = Annotation(
    type=pd.Series,
    title='Task weights',
    description='A pandas.Series indexed by `task` containing task weights'
)


def _make_opitonal(annotation: Annotation):
    return attr.evolve(annotation, type=Optional[annotation.type])


OPTIONAL_SCORES = _make_opitonal(TASKS_LABEL_SCORES)
OPTIONAL_SKILLS = _make_opitonal(SKILLS)
OPTIONAL_PROBAS = _make_opitonal(TASKS_LABEL_PROBAS)
OPTIONAL_PRIORS = _make_opitonal(LABEL_PRIORS)
OPTIONAL_LABELS = _make_opitonal(TASKS_LABELS)
OPTIONAL_ERRORS = _make_opitonal(ERRORS)
OPTIONAL_WEIGHTS = _make_opitonal(WEIGHTS)
