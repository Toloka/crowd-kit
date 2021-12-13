__all__ = [
    'BaseClassificationAggregator',
    'BaseImageSegmentationAggregator',
    'BaseEmbeddingsAggregator',
    'BaseTextsAggregator',
    'BasePairwiseAggregator',
]

import attr

from .. import annotations
from ..utils import named_series_attrib


@attr.s
@annotations.manage_docstring
class BaseClassificationAggregator:
    """ This is a base class for all classification aggregators"""

    labels_: annotations.OPTIONAL_LABELS = named_series_attrib(name='agg_label')

    @annotations.manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> annotations.Annotation(type='BaseClassificationAggregator',
                                                                            title='self'):
        raise NotImplementedError()

    @annotations.manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        raise NotImplementedError()


@attr.s
@annotations.manage_docstring
class BaseImageSegmentationAggregator:
    """This is a base class for all image segmentation aggregators"""

    segmentations_: annotations.TASKS_SEGMENTATIONS = named_series_attrib(name='agg_segmentation')

    @annotations.manage_docstring
    def fit(self, data: annotations.SEGMENTATION_DATA) -> annotations.Annotation(type='BaseImageSegmentationAggregator',
                                                                                 title='self'):
        raise NotImplementedError()

    @annotations.manage_docstring
    def fit_predict(self, data: annotations.SEGMENTATION_DATA) -> annotations.TASKS_SEGMENTATIONS:
        raise NotImplementedError()


@attr.s
@annotations.manage_docstring
class BaseEmbeddingsAggregator:
    """This is a base class for all embeddings aggregators"""

    embeddings_and_outputs_: annotations.TASKS_EMBEDDINGS_AND_OUTPUTS = attr.ib(init=False)

    @annotations.manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA) -> annotations.Annotation(type='BaseEmbeddingsAggregator', title='self'):
        raise NotImplementedError()

    @annotations.manage_docstring
    def fit_predict(self, data: annotations.EMBEDDED_DATA) -> annotations.TASKS_EMBEDDINGS_AND_OUTPUTS:
        raise NotImplementedError()


@attr.s
@annotations.manage_docstring
class BaseTextsAggregator:
    """ This is a base class for all texts aggregators"""

    texts_: annotations.TASKS_TEXTS = named_series_attrib(name='agg_text')

    @annotations.manage_docstring
    def fit(self, data: annotations.TEXT_DATA) -> annotations.Annotation(type='BaseTextsAggregator', title='self'):
        raise NotImplementedError()

    @annotations.manage_docstring
    def fit_predict(self, data: annotations.TEXT_DATA) -> annotations.TASKS_TEXTS:
        raise NotImplementedError()


@attr.s
@annotations.manage_docstring
class BasePairwiseAggregator:
    """ This is a base class for all pairwise comparison aggregators"""

    scores_: annotations.LABEL_SCORES = named_series_attrib(name='agg_score')

    @annotations.manage_docstring
    def fit(self, data: annotations.PAIRWISE_DATA) -> annotations.Annotation(type='BasePairwiseAggregator', title='self'):
        raise NotImplementedError()

    @annotations.manage_docstring
    def fit_predict(self, data: annotations.PAIRWISE_DATA) -> annotations.LABEL_SCORES:
        raise NotImplementedError()
