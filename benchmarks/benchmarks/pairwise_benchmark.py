from crowdkit.datasets import load_dataset
from crowdkit.aggregation import BradleyTerry, NoisyBradleyTerry


class IMDBWikiSbs:

    def setup(self):
        self.crowd_data, self.ground_truth = load_dataset('idmb-wiki-sbs')

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

    # TODO: add ndcg_score after dataset fix
