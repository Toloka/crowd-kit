"""
This module contains reusable annotations that encapsulate both typing
and description for commonly used parameters. These annotations are
used to automatically generate stub files with proper docstrings
"""
__all__ = [
    'Annotation',
    'manage_docstring',

    'DATA',
    'EMBEDDED_DATA',
    'LABELED_DATA',
    'PAIRWISE_DATA',
    'TEXT_DATA',
    'SEGMENTATION_DATA',

    'LABEL_PRIORS',
    'LABEL_SCORES',
    'TASKS_EMBEDDINGS',
    'TASKS_LABELS',
    'TASKS_LABEL_PROBAS',
    'TASKS_LABEL_SCORES',
    'TASKS_TRUE_LABELS',
    'SEGMENTATIONS',
    'SEGMENTATION',
    'SEGMENTATION_ERRORS',
    'IMAGE_PIXEL_PROBAS',
    'TASKS_SEGMENTATIONS',
    'TASKS_TEXTS',
    'GLAD_ALPHAS',
    'GLAD_BETAS',
    'BIASES',
    'SKILLS',
    'ERRORS',
    'WEIGHTED_DATA',
    'WEIGHTS',
    'ON_MISSING_SKILL',

    'OPTIONAL_ERRORS',
    'OPTIONAL_PRIORS',
    'OPTIONAL_PROBAS',
    'OPTIONAL_SCORES',
    'OPTIONAL_LABELS',
    'OPTIONAL_SKILLS',
    'OPTIONAL_WEIGHTS',
    'OPTIONAL_TEXTS',
]

import inspect
import textwrap
from copy import deepcopy
from io import StringIO
from typing import Dict, Optional, Type

import attr
import numpy as np
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

    traverse_order = [obj]

    if isinstance(obj, type):
        traverse_order = list(obj.__mro__[::-1])

    for cur_obj in traverse_order:

        if cur_obj != obj:
            annotations = getattr(cur_obj, '_orig_annotations', getattr(cur_obj, '__annotations__', {}))
        else:
            annotations = getattr(cur_obj, '__annotations__', {})
        for key, value in annotations.items():
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

    if isinstance(obj, type) and not hasattr(obj, '_orig_annotations'):
        obj._orig_annotations = deepcopy(getattr(obj, '__annotations__', {}))

    obj.__annotations__ = new_annotations
    obj.__doc__ = sio.getvalue()
    return obj


# Input data descriptions


EMBEDDED_DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' outputs with their embeddings",
    description='A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.'
)

DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' outputs",
    description='A pandas.DataFrame containing `task`, `performer` and `output` columns.'
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
        For each row `label` must be equal to either `left` column or `right` column.
    ''')
)

TEXT_DATA = Annotation(
    type=pd.DataFrame,
    title="Performers' text outputs",
    description='A pandas.DataFrame containing `task`, `performer` and `text` columns.'
)


SEGMENTATION_DATA = Annotation(
    type=pd.DataFrame,
    title='Performers\' segmentations',
    description=textwrap.dedent('''
        A pandas.DataFrame containing `performer`, `task` and `segmentation` columns'.
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
    type=pd.Series,
    title="Tasks' embeddings",
    description=textwrap.dedent("A pandas.Series indexed by `task` and holding corresponding embeddings."),
)

TASKS_LABELS = Annotation(
    type=pd.Series,
    title="Tasks' labels",
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` such that `labels.loc[task]`
        is the tasks's most likely true label.
    '''),
)


TASKS_EMBEDDINGS_AND_OUTPUTS = Annotation(
    type=pd.DataFrame,
    title="Tasks' embeddings and outputs",
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
    '''),
)

TASKS_LABEL_PROBAS = Annotation(
    type=pd.DataFrame,
    title="Tasks' label probability distributions",
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
        is the probability of `task`'s true label to be equal to `label`. Each
        probability is between 0 and 1, all task's probabilities should sum up to 1
    '''),
)

TASKS_LABEL_SCORES = Annotation(
    type=pd.DataFrame,
    title="Tasks' label scores",
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

SEGMENTATIONS = Annotation(
    type=pd.Series,
    title='Single task segmentations',
    description=textwrap.dedent('''
         A pandas.Series containing segmentations for a single task - 2d boolean ndarray.
    '''),
)

SEGMENTATION = Annotation(
    type=np.ndarray,
    title='Segmentation',
    description=textwrap.dedent('''
         A numpy 2d ndarray, which is a bitmask of segmentation.
    '''),
)

SEGMENTATION_ERRORS = Annotation(
    type=np.ndarray,
    title='Errors',
    description=textwrap.dedent('''
         A numpy 1d ndarray, which contains the probability of correct answers for performers.
    '''),
)

IMAGE_PIXEL_PROBAS = Annotation(
    type=np.ndarray,
    title='Image pixel probas',
    description=textwrap.dedent('''
         A numpy 2d ndarray, which in each pixel contains the probability of inclusion in the segmentation.
    '''),
)

TASKS_SEGMENTATIONS = Annotation(
    type=pd.Series,
    title='Tasks\' segmentations',
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` such that `labels.loc[task]`
        is the tasks's aggregated segmentation.
    '''),
)

TASKS_TEXTS = Annotation(
    type=pd.Series,
    title="Tasks' texts",
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` such that `result.loc[task, text]`
        is the task's text.
    '''),
)

GLAD_ALPHAS = Annotation(
    type=pd.Series,
    title="Performers' alpha parameters",
    description=textwrap.dedent('''
        A pandas.Series indexed by `performer` that contains estimated alpha parameters.
    '''),
)

GLAD_BETAS = Annotation(
    type=pd.Series,
    title="Tasks' beta parameters",
    description=textwrap.dedent('''
        A pandas.Series indexed by `task` that contains estimated beta parameters.
    '''),
)


# Performers related annotations

BIASES = Annotation(
    type=pd.Series,
    title='Predicted biases for each performer. Indicates the probability of a performer to choose the left item.',
    description=textwrap.dedent("A series of performers' biases indexed by performers"),
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
    type=pd.DataFrame,
    title='Task weights',
    description='A pandas.DataFrame containing `task`, `performer` and `weight`'
)


# Common parameters

ON_MISSING_SKILL = Annotation(
    type=str,
    title="How to handle assignments done by workers with unknown skill",
    description="Possible values:\n"
    '\t* "error" — raise an exception if there is at least one assignment done by user with unknown skill;\n'
    '\t* "ignore" — drop assignments with unknown skill values during prediction. Raise an exception if there is no \n'
    '\tassignments with known skill for any task;\n'
    '\t* value — default value will be used if skill is missing.'
)


def _make_optional(annotation: Annotation):
    return attr.evolve(annotation, type=Optional[annotation.type])


OPTIONAL_SCORES = _make_optional(TASKS_LABEL_SCORES)
OPTIONAL_SKILLS = _make_optional(SKILLS)
OPTIONAL_PROBAS = _make_optional(TASKS_LABEL_PROBAS)
OPTIONAL_PRIORS = _make_optional(LABEL_PRIORS)
OPTIONAL_LABELS = _make_optional(TASKS_LABELS)
OPTIONAL_ERRORS = _make_optional(ERRORS)
OPTIONAL_WEIGHTS = _make_optional(WEIGHTS)
OPTIONAL_TEXTS = _make_optional(TASKS_TEXTS)
