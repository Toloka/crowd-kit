import numpy as np

from crowdkit.datasets import load_dataset
from crowdkit.aggregation import SegmentationEM, SegmentationRASA, SegmentationMajorityVote


class MSCOCOSmall:

    timeout = 600

    def setup(self):
        self.crowd_segmentations, self.ground_truth = load_dataset('mscoco_small')

    def time_segmentation_em(self):
        SegmentationEM(n_iter=1).fit_predict(self.crowd_segmentations)

    def time_segmentation_rasa(self):
        SegmentationRASA(n_iter=1).fit_predict(self.crowd_segmentations)

    def time_segmentation_mv(self):
        SegmentationMajorityVote().fit_predict(self.crowd_segmentations)

    # peakmem

    def peakmem_segmentation_em(self):
        SegmentationEM(n_iter=1).fit_predict(self.crowd_segmentations)

    def peakmem_segmentation_rasa(self):
        SegmentationRASA(n_iter=1).fit_predict(self.crowd_segmentations)

    def peakmem_segmentation_mv(self):
        SegmentationMajorityVote().fit_predict(self.crowd_segmentations)

    # accuracy

    def _calc_accuracy(self, predict):
        acc = 0
        shapes = 0
        for task, gt_task in zip(predict, self.ground_truth):
            acc += np.sum(task == gt_task)
            shapes += task.size
        return acc / shapes

    def track_accuracy_segmentation_em(self):
        return self._calc_accuracy(SegmentationEM(n_iter=1).fit_predict(self.crowd_segmentations))

    def track_accuracy_segmentation_rasa(self):
        return self._calc_accuracy(SegmentationRASA(n_iter=1).fit_predict(self.crowd_segmentations))

    def track_accuracy_segmentation_mv(self):
        return self._calc_accuracy(SegmentationMajorityVote().fit_predict(self.crowd_segmentations))
