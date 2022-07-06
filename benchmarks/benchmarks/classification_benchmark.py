from crowdkit.datasets import load_dataset
from crowdkit.aggregation import GoldMajorityVote, MajorityVote, DawidSkene, MMSR, Wawa, ZeroBasedSkill, GLAD
from crowdkit.aggregation.utils import get_accuracy


class NistTrecRelevance:
    def setup(self):
        self.crowd_labels, self.ground_truth = load_dataset('nist-trec-relevance')

    def time_gold_majority_vote(self):
        GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth)

    def time_majority_vote(self):
        MajorityVote().fit_predict(self.crowd_labels)

    def time_dawid_skene(self):
        DawidSkene(n_iter=5).fit_predict(self.crowd_labels)

    def time_mmsr(self):
        MMSR(n_iter=5).fit_predict(self.crowd_labels)

    def time_wawa(self):
        Wawa().fit_predict(self.crowd_labels)

    def time_zbs(self):
        ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels)

    def time_glad(self):
        GLAD(max_iter=5).fit_predict(self.crowd_labels)

    # peak memory

    def peakmem_gold_majority_vote(self):
        GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth)

    def peakmem_majority_vote(self):
        MajorityVote().fit_predict(self.crowd_labels)

    def peakmem_dawid_skene(self):
        DawidSkene(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_mmsr(self):
        MMSR(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_wawa(self):
        Wawa().fit_predict(self.crowd_labels)

    def peakmem_zbs(self):
        ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_glad(self):
        GLAD(max_iter=5).fit_predict(self.crowd_labels)

    # accuracy

    def _calc_accuracy(self, predict):
        predict = predict.to_frame().reset_index()
        predict.columns = ['task', 'label']
        predict['worker'] = None
        return get_accuracy(predict, true_labels=self.ground_truth)

    def track_accuracy_gold_majority_vote(self):
        return self._calc_accuracy(GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth))

    def track_accuracy_majority_vote(self):
        return self._calc_accuracy(MajorityVote().fit_predict(self.crowd_labels))

    def track_accuracy_dawid_skene(self):
        return self._calc_accuracy(DawidSkene(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_mmsr(self):
        return self._calc_accuracy(MMSR(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_wawa(self):
        return self._calc_accuracy(Wawa().fit_predict(self.crowd_labels))

    def track_accuracy_zbs(self):
        return self._calc_accuracy(ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_glad(self):
        return self._calc_accuracy(GLAD(max_iter=5).fit_predict(self.crowd_labels))
