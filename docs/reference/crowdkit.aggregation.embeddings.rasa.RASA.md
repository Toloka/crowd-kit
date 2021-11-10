# RASA
`crowdkit.aggregation.embeddings.rasa.RASA`

```
RASA(
    self,
    n_iter: int = 100,
    alpha: float = ...
)
```

Reliability Aware Sequence Aggregation


Jiyi Li. 2019.
A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP,
pages 24–28 Hong Kong, China, November 3, 2019.
http://doi.org/10.18653/v1/D19-5904

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`embeddings_and_outputs_`|**DataFrame**|<p>Tasks&#x27; embeddings and outputs A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.embeddings.rasa.RASA.fit.md)| None
[fit_predict](crowdkit.aggregation.embeddings.rasa.RASA.fit_predict.md)| None
[fit_predict_scores](crowdkit.aggregation.embeddings.rasa.RASA.fit_predict_scores.md)| None
