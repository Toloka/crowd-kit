__all__ = ['SegmentationRASA']

import typing

import attr
import numpy as np

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseImageSegmentationAggregator


_EPS = 1e-5


@attr.s
@manage_docstring
class SegmentationRASA(BaseImageSegmentationAggregator):
    """
    Segmentation RASA - chooses a pixel if sum of weighted votes of each workers' more than 0.5.

    Algorithm works iteratively, at each step, the workers are reweighted in proportion to their distances
    to the current answer estimation. The distance is considered as $1 - IOU$. Modification of the RASA method
    for texts.

    Jiyi Li.
    A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
    *Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*,
    pages 24–28 Hong Kong, China, November 3, 2019.
    <https://doi.org/10.18653/v1/D19-5904>

    Args:
        n_iter: A number of iterations.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from crowdkit.aggregation import SegmentationRASA
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', np.array([[1, 0], [1, 1]])],
        >>>         ['t1', 'p2', np.array([[0, 1], [1, 1]])],
        >>>         ['t1', 'p3', np.array([[0, 1], [1, 1]])]
        >>>     ],
        >>>     columns=['task', 'worker', 'segmentation']
        >>> )
        >>> result = SegmentationRASA().fit_predict(df)
    """

    n_iter: int = attr.ib(default=10)
    tol: float = attr.ib(default=1e-5)
    # segmentations_
    loss_history_: typing.List[float] = attr.ib(init=False)

    @staticmethod
    @manage_docstring
    def _segmentation_weighted(segmentations: annotations.SEGMENTATIONS, weights: annotations.SEGMENTATION_ERRORS) -> annotations.SEGMENTATION:
        """
        Performs weighted majority vote algorithm.

        From the weights of all workers and their segmentation, performs a
        weighted majority vote for the inclusion of each pixel in the answer.
        """
        weighted_segmentations = (weights * segmentations.T).T
        return weighted_segmentations.sum(axis=0)

    @staticmethod
    @manage_docstring
    def _calculate_weights(segmentations: annotations.SEGMENTATIONS, mv: annotations.SEGMENTATION) -> annotations.SEGMENTATION_ERRORS:
        """
        Calculates weights of each workers, from current majority vote estimation.
        """
        intersection = (segmentations & mv).astype(float)
        union = (segmentations | mv).astype(float)
        distances = 1 - intersection.sum(axis=(1, 2))/union.sum(axis=(1, 2))
        # add a small bias for more
        # numerical stability and correctness of transform.
        weights = np.log(1 / (distances + _EPS) + 1)
        return weights / np.sum(weights)

    @manage_docstring
    def _aggregate_one(self, segmentations: annotations.SEGMENTATIONS) -> annotations.SEGMENTATION:
        """
        Performs Segmentation RASA algorithm for a single image.
        """
        size = len(segmentations)
        segmentations = np.stack(segmentations.values)
        weights = np.full(size, 1 / size)
        mv = self._segmentation_weighted(segmentations, weights)

        last_aggregated = None
        self.loss_history_ = []
        for _ in range(self.n_iter):
            weighted = self._segmentation_weighted(segmentations, weights)
            mv = weighted >= 0.5
            weights = self._calculate_weights(segmentations, mv)
            if last_aggregated is not None:
                delta = weighted - last_aggregated
                loss = (delta * delta).sum().sum() / (weighted * weighted).sum().sum()
                self.loss_history_.append(loss)
                if loss < self.tol:
                    break
            last_aggregated = weighted
        return mv

    @manage_docstring
    def fit(self, data: annotations.SEGMENTATION_DATA) -> Annotation(type='SegmentationRASA', title='self'):
        """
        Fit the model.
        """

        data = data[['task', 'worker', 'segmentation']]

        # The latest pandas version installable under Python3.7 is pandas 1.1.5.
        # This version fails to accept a method with an error but works fine with lambdas
        # >>> TypeError: unhashable type: 'SegmentationRASA'duito an inner logic that tries
        aggregate_one = lambda arg: self._aggregate_one(arg)

        self.segmentations_ = data.groupby('task').segmentation.apply(aggregate_one)
        return self

    @manage_docstring
    def fit_predict(self, data: annotations.SEGMENTATION_DATA) -> annotations.TASKS_SEGMENTATIONS:
        """
        Fit the model and return the aggregated segmentations.
        """

        return self.fit(data).segmentations_
