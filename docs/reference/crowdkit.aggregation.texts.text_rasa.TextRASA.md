# TextRASA
`crowdkit.aggregation.texts.text_rasa.TextRASA` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/text_rasa.py#L9)

```python
TextRASA(
    self,
    encoder: Callable,
    n_iter: int = 100,
    tol: float = 1e-05,
    alpha: float = 0.05
)
```

RASA on text embeddings.


Given a sentence encoder, encodes texts provided by workers and runs the RASA algorithm for embedding
aggregation.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`encoder`|**Callable**|<p>A callable that takes a text and returns a NumPy array containing the corresponding embedding.</p>
`n_iter`|**int**|<p>A number of RASA iterations.</p>
`alpha`|**float**|<p>Confidence level of chi-squared distribution quantiles in beta parameter formula.</p>

**Examples:**

We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
```python
from crowdkit.datasets import load_dataset
from crowdkit.aggregation import TextRASA
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-mpnet-base-v2')
hrrasa = TextRASA(encoder=encoder.encode)
df, gt = load_dataset('crowdspeech-test-clean')
df['text'] = df['text'].apply(lambda s: s.lower())
result = hrrasa.fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.texts.text_rasa.TextRASA.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.texts.text_rasa.TextRASA.fit_predict.md)| Fit the model and return aggregated texts.
[fit_predict_scores](crowdkit.aggregation.texts.text_rasa.TextRASA.fit_predict_scores.md)| Fit the model and return scores.
