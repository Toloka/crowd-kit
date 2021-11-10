from crowdkit.datasets import load_dataset
from crowdkit.aggregation import TextRASA, TextHRRASA, ROVER
from jiwer import wer
from sklearn.feature_extraction.text import TfidfVectorizer


class CrowdspeechTestClean:
    timeout = 180

    def setup(self):
        self.crowd_texts, self.ground_truth = load_dataset('crowdspeech-test-clean')
        self.transformer = TfidfVectorizer(max_features=1000).fit(self.crowd_texts['text'].tolist())

    def tokenizer(self, text):
        return text.split(' ')

    def detokenizer(self, tokens):
        return ' '.join(tokens)

    def encoder(self, text):
        return self.transformer.transform([text]).toarray()

    # time

    def time_text_hrrasa(self):
        TextHRRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))

    def time_text_rasa(self):
        TextRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))

    def time_text_rover(self):
        ROVER(tokenizer=self.tokenizer, detokenizer=self.detokenizer).fit_predict(self.crowd_texts)

    # peakmem

    def peakmem_text_hrrasa(self):
        TextHRRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))

    def peakmem_text_rasa(self):
        TextRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))

    def peakmem_text_rover(self):
        ROVER(tokenizer=self.tokenizer, detokenizer=self.detokenizer).fit_predict(self.crowd_texts)

    # accuracy

    def _calc_wer(self, predict):
        gt_list, pred_list = [], []
        for task, text in self.ground_truth.iteritems():
            gt_list.append(text)
            pred_list.append(predict.loc[task])
        return wer(gt_list, pred_list)

    def track_wer_text_rasa(self):
        prediction = TextRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))
        return self._calc_wer(prediction)

    def track_wer_text_hrrasa(self):
        prediction = TextHRRASA(encoder=self.encoder, n_iter=5).fit_predict(self.crowd_texts.rename(columns={'text': 'output'}))
        return self._calc_wer(prediction)

    def track_wer_rover(self):
        prediction = ROVER(tokenizer=self.tokenizer, detokenizer=self.detokenizer).fit_predict(self.crowd_texts)
        return self._calc_wer(prediction)
