__all__ = ['SegmentationMajorityVote']

from typing import Optional

import attr
import numpy as np
import pandas as pd

from ..base import BaseImageSegmentationAggregator
from ..utils import add_skills_to_data


@attr.s
class SegmentationMajorityVote(BaseImageSegmentationAggregator):
    """Segmentation Majority Vote - chooses a pixel if more than half of workers voted.

    This method implements a straightforward approach to the image segmentations aggregation:
    it assumes that if pixel is not inside in the worker's segmentation, this vote counts
    as 0, otherwise, as 1. Next, the `SegmentationEM` aggregates these categorical values
    for each pixel by the Majority Vote.

    The method also supports weighted majority voting if `skills` were provided to `fit` method.

    Doris Jung-Lin Lee. 2018.
    Quality Evaluation Methods for Crowdsourced Image Segmentation
    <https://ilpubs.stanford.edu:8090/1161/1/main.pdf>

    Args:
        default_skill: A default skill value for missing skills.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from crowdkit.aggregation import SegmentationMajorityVote
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', np.array([[1, 0], [1, 1]])],
        >>>         ['t1', 'p2', np.array([[0, 1], [1, 1]])],
        >>>         ['t1', 'p3', np.array([[0, 1], [1, 1]])]
        >>>     ],
        >>>     columns=['task', 'worker', 'segmentation']
        >>> )
        >>> result = SegmentationMajorityVote().fit_predict(df)

    Attributes:
        segmentations_ (Series): Tasks' segmentations.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's aggregated segmentation.

        on_missing_skill (str): How to handle assignments done by workers with unknown skill.
            Possible values:
                    * "error" — raise an exception if there is at least one assignment done by user with unknown skill;
                    * "ignore" — drop assignments with unknown skill values during prediction. Raise an exception if there is no
                    assignments with known skill for any task;
                    * value — default value will be used if skill is missing.
    """

    # segmentations_

    on_missing_skill: str = attr.ib(default='error')
    default_skill: Optional[float] = attr.ib(default=None)

    def fit(self, data: pd.DataFrame, skills: pd.Series = None) -> 'SegmentationMajorityVote':
        """
        Fit the model.
        """

        data = data[['task', 'worker', 'segmentation']]

        if skills is None:
            data['skill'] = 1
        else:
            data = add_skills_to_data(data, skills, self.on_missing_skill, self.default_skill)

        data['pixel_scores'] = data.segmentation * data.skill
        group = data.groupby('task')

        self.segmentations_ = (2 * group.pixel_scores.apply(np.sum) - group.skill.apply(np.sum)).apply(lambda x: x >= 0)
        return self

    def fit_predict(self, data: pd.DataFrame, skills: Optional[pd.Series] = None) -> pd.Series:
        """
        Fit the model and return the aggregated segmentations.
        """

        return self.fit(data, skills).segmentations_
