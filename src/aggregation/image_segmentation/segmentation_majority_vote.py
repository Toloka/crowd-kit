__all__ = ['SegmentationMajorityVote']
import attr
import numpy as np

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseImageSegmentationAggregator


@attr.s
@manage_docstring
class SegmentationMajorityVote(BaseImageSegmentationAggregator):
    """
    Majority Vote - chooses a pixel if more than half of performers voted

    Doris Jung-Lin Lee. 2018.
    Quality Evaluation Methods for Crowdsourced Image Segmentation
    http://ilpubs.stanford.edu:8090/1161/1/main.pdf

    """

    # segmentations_

    @manage_docstring
    def fit(self, data: annotations.SEGMENTATION_DATA, skills: annotations.SKILLS = None) -> Annotation(type='SegmentationMajorityVote', title='self'):
        data = data[['task', 'performer', 'segmentation']]

        if skills is None:
            data['skill'] = 1
        else:
            data = data.join(skills.rename('skill'), on='performer')

        data['pixel_scores'] = data.segmentation * data.skill
        group = data.groupby('task')

        self.segmentations_ = (2 * group.pixel_scores.apply(np.sum) - group.skill.apply(np.sum)).apply(lambda x: x >= 0)
        return self

    @manage_docstring
    def fit_predict(self, data: annotations.SEGMENTATION_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_SEGMENTATIONS:
        return self.fit(data, skills).segmentations_
