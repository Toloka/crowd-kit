# TextHRRASA
`crowdkit.aggregation.texts.text_hrrasa.TextHRRASA`

```python
TextHRRASA(
    self,
    encoder: Callable,
    n_iter: int = 100,
    tol: float = 1e-05,
    lambda_emb: float = 0.5,
    lambda_out: float = 0.5,
    alpha: float = 0.05,
    calculate_ranks: bool = False,
    output_similarity: Callable = glue_similarity
)
```

HRRASA on text embeddings.


Given a sentence encoder, encodes texts provided by performers and runs the HRRASA algorithm for embedding
aggregation.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`encoder`|**Callable**|<p>A callable that takes a text and returns a NumPy array containing the corresponding embedding.</p>
`n_iter`|**int**|<p>A number of HRRASA iterations.</p>
`lambda_emb`|**float**|<p>A weight of reliability calculated on embeddigs.</p>
`lambda_out`|**float**|<p>A weight of reliability calculated on outputs.</p>
`alpha`|**float**|<p>Confidence level of chi-squared distribution quantiles in beta parameter formula.</p>
`calculate_ranks`|**bool**|<p>If true, calculate additional attribute `ranks_`.</p>

**Examples:**

We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
```python
from crowdkit.datasets import load_dataset
from crowdkit.aggregation import TextHRRASA
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-mpnet-base-v2')
hrrasa = TextHRRASA(encoder=encoder.encode)
df, gt = load_dataset('crowdspeech-test-clean')
df['text'] = df['text'].apply(lambda s: s.lower())
result = hrrasa.fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit_predict](crowdkit.aggregation.texts.text_hrrasa.TextHRRASA.fit_predict.md)| Fit the model and return aggregated texts.
[fit_predict_scores](crowdkit.aggregation.texts.text_hrrasa.TextHRRASA.fit_predict_scores.md)| Fit the model and return scores.
