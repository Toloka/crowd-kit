"""
This module contains reusable annotations that encapsulate both typing
and description for commonly used parameters. These annotations are
used to automatically generate stub files with proper docstrings
"""

import inspect
import textwrap
from io import StringIO
from typing import ClassVar, Dict, Optional, Type, get_type_hints

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

    for key, value in get_type_hints(obj).items():
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


PERFORMERS_SKILLS = Annotation(
    type=pd.Series,
    title='Predicted skills for each performer',
    description=textwrap.dedent("A series of performers' skills indexed by performers"),
)

PROBAS = Annotation(
    type=pd.DataFrame,
    title='Estimated label probabilities',
    description=textwrap.dedent('''
        A frame indexed by `task` and a column for every label id found
        in `data` such that `result.loc[task, label]` is the probability of `task`'s
        true label to be equal to `label`.
    '''),
)

PRIORS = Annotation(
    type=pd.Series,
    title='A prior label distribution',
    description="A series of labels' probabilities indexed by labels",
)

TASKS_LABELS = Annotation(
    type=pd.DataFrame,
    title='Estimated labels',
    description=textwrap.dedent('''
        A pandas.DataFrame indexed by `task` with a single column `label` containing
        `tasks`'s most probable label for last fitted data, or None otherwise.
    '''),
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

DATA = Annotation(
    type=pd.DataFrame,
    title='Input data',
    description='A pandas.DataFrame containing `task`, `performer` and `label` columns',
)


def _make_opitonal_classlevel(annotation: Annotation):
    return attr.evolve(annotation, type=ClassVar[Optional[annotation.type]])


OPTIONAL_CLASSLEVEL_PERFORMERS_SKILLS = _make_opitonal_classlevel(PERFORMERS_SKILLS)
OPTIONAL_CLASSLEVEL_PROBAS = _make_opitonal_classlevel(PROBAS)
OPTIONAL_CLASSLEVEL_PRIORS = _make_opitonal_classlevel(PRIORS)
OPTIONAL_CLASSLEVEL_TASKS_LABELS = _make_opitonal_classlevel(TASKS_LABELS)
OPTIONAL_CLASSLEVEL_ERRORS = _make_opitonal_classlevel(ERRORS)
