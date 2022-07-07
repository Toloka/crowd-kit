from crowdkit.datasets import load_dataset
from crowdkit.aggregation import BradleyTerry, NoisyBradleyTerry
from sklearn.metrics import ndcg_score

import numpy as np


class IMDBWikiSbs:

    def setup(self):
        self.crowd_data, self.ground_truth = load_dataset('imdb-wiki-sbs')

    # time

    def time_bt(self):
        BradleyTerry(n_iter=5).fit_predict(self.crowd_data)

    def time_noisy_bt(self):
        NoisyBradleyTerry(n_iter=5).fit_predict(self.crowd_data)

    # peakmem

    def peakmem_bt(self):
        BradleyTerry(n_iter=5).fit_predict(self.crowd_data)

    def peakmem_noisy_bt(self):
        NoisyBradleyTerry(n_iter=5).fit_predict(self.crowd_data)

    # track_ndcg_score

    def _calc_ndcg_score(self, pred):
        y_true, y_pred = [], []
        for task, true_value in self.ground_truth.iteritems():
            y_true.append(true_value)
            y_pred.append(pred.loc[task])
        return ndcg_score(np.array(y_pred).reshape(1, -1), np.array(y_true).reshape(1, -1), k=10)

    def track_ndcg_score_bt(self):
        return self._calc_ndcg_score(BradleyTerry(n_iter=5).fit_predict(self.crowd_data))

    def track_ndcg_score_noisy_bt(self):
        return self._calc_ndcg_score(NoisyBradleyTerry(n_iter=5).fit_predict(self.crowd_data))
