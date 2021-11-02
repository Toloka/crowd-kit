__all__ = ['SegmentationEM']

import attr
import numpy as np

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseImageSegmentationAggregator


@attr.s
@manage_docstring
class SegmentationEM(BaseImageSegmentationAggregator):
    """
    The EM algorithm for the image segmentation task.
    For each task, EM algorithm is performed to classify the image pixels.
    Doris Jung-Lin Lee. 2018.
    Quality Evaluation Methods for Crowdsourced Image Segmentation
    http://ilpubs.stanford.edu:8090/1161/1/main.pdf
    """

    n_iter: int = attr.ib(default=10)
    # segmentations_

    @staticmethod
    @manage_docstring
    def _e_step(
        segmentations: annotations.SEGMENTATIONS,
        errors: annotations.SEGMENTATION_ERRORS,
        priors: annotations.IMAGE_PIXEL_PROBAS,
    ) -> annotations.IMAGE_PIXEL_PROBAS:
        """
        Perform E-step of algorithm.
        Given performers' segmentations and error vector and priors
        for each pixel calculates posteriori probabilities.
        """

        weighted_seg = np.multiply(errors, segmentations.T.astype(float)).T +\
                        np.multiply((1 - errors), (1 - segmentations).T.astype(float)).T

        with np.errstate(divide='ignore'):
            pos_log_prob = np.log(priors) + np.log(weighted_seg).sum(axis=0)
            neg_log_prob = np.log(1 - priors) + np.log(1 - weighted_seg).sum(axis=0)

            with np.errstate(invalid='ignore'):
                # division by the denominator in the Bayes formula
                priors = np.nan_to_num(np.exp(pos_log_prob) / (np.exp(pos_log_prob) + np.exp(neg_log_prob)), nan=0)

        return priors

    @staticmethod
    @manage_docstring
    def _m_step(
        segmentations: annotations.SEGMENTATIONS,
        priors: annotations.IMAGE_PIXEL_PROBAS,
        segmentation_region_size: int,
        segmentations_sizes: np.ndarray
    ) -> annotations.SEGMENTATION_ERRORS:
        """
        Perform M-step of algorithm.
        Given a priori probabilities for each pixel and the segmentation of the performers,
        it estimates performer's errors probabilities vector.
        """

        mean_errors_expectation = (segmentations_sizes + priors.sum() -
                                   2 * (segmentations * priors).sum(axis=(1, 2))) / segmentation_region_size

        # return probability of worker marking pixel correctly
        return 1 - mean_errors_expectation

    @manage_docstring
    def _aggregate_one(self, segmentations: annotations.SEGMENTATIONS) -> annotations.SEGMENTATION:
        """
        Performs an expectation maximization algorithm for a single image.
        """
        priors = sum(segmentations) / len(segmentations)
        segmentations = np.stack(segmentations.values)
        segmentation_region_size = segmentations.any(axis=0).sum()
        if segmentation_region_size == 0:
            return np.zeros_like(segmentations[0])

        segmentations_sizes = segmentations.sum(axis=(1, 2))
        # initialize with errors assuming that ground truth segmentation is majority vote
        errors = self._m_step(segmentations, np.round(priors), segmentation_region_size, segmentations_sizes)
        for _ in range(self.n_iter):
            priors = self._e_step(segmentations, errors, priors)
            errors = self._m_step(segmentations, priors, segmentation_region_size, segmentations_sizes)
        return priors > 0.5

    @manage_docstring
    def fit(self, data: annotations.SEGMENTATION_DATA) -> Annotation(type='SegmentationEM', title='self'):
        data = data[['task', 'performer', 'segmentation']]

        self.segmentations_ = data.groupby('task').segmentation.apply(
            lambda segmentations: self._aggregate_one(segmentations)  # using lambda for python 3.7 compatibility
        )
        return self

    @manage_docstring
    def fit_predict(self, data: annotations.SEGMENTATION_DATA) -> annotations.TASKS_SEGMENTATIONS:
        return self.fit(data).segmentations_
