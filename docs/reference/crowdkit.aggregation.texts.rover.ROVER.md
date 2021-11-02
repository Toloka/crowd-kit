# ROVER
`crowdkit.aggregation.texts.rover.ROVER`

```
ROVER(
    self,
    tokenizer: Callable[[str], List[str]],
    detokenizer: Callable[[List[str]], str],
    silent: bool = True
)
```

Recognizer Output Voting Error Reduction (ROVER)


J. G. Fiscus,
"A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction (ROVER),"
1997 IEEE Workshop on Automatic Speech Recognition and Understanding Proceedings, 1997, pp. 347-354.
https://doi.org/10.1109/ASRU.1997.659110

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`texts_`|**Series**|<p>Tasks&#x27; texts A pandas.Series indexed by `task` such that `result.loc[task, text]` is the task&#x27;s text.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.texts.rover.ROVER.fit.md)| None
[fit_predict](crowdkit.aggregation.texts.rover.ROVER.fit_predict.md)| None
