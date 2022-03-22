# ROVER
`crowdkit.aggregation.texts.rover.ROVER` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/rover.py#L31)

```python
ROVER(
    self,
    tokenizer: Callable[[str], List[str]],
    detokenizer: Callable[[List[str]], str],
    silent: bool = True
)
```

Recognizer Output Voting Error Reduction (ROVER).


This method uses dynamic programming to align sequences. Next, aligned sequences are used
to construct the Word Transition Network (WTN):
![ROVER WTN scheme](http://tlk.s3.yandex.net/crowd-kit/docs/rover.png)
Finally, the aggregated sequence is the result of majority voting on each edge of the WTN.

J. G. Fiscus,
"A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction (ROVER),"
*1997 IEEE Workshop on Automatic Speech Recognition and Understanding Proceedings*, 1997, pp. 347-354.
https://doi.org/10.1109/ASRU.1997.659110

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`tokenizer`|**Callable\[\[str\], List\[str\]\]**|<p>A callable that takes a string and returns a list of tokens.</p>
`detokenizer`|**Callable\[\[List\[str\]\], str\]**|<p>A callable that takes a list of tokens and returns a string.</p>
`silent`|**bool**|<p>If false, show a progress bar.</p>
`texts_`|**Series**|<p>Tasks&#x27; texts. A pandas.Series indexed by `task` such that `result.loc[task, text]` is the task&#x27;s text.</p>

**Examples:**

```python
from crowdkit.aggregation import load_dataset
from crowdkit.aggregation import ROVER
df, gt = load_dataset('crowdspeech-test-clean')
df['text'] = df['text].apply(lambda s: s.lower())
tokenizer = lambda s: s.split(' ')
detokenizer = lambda tokens: ' '.join(tokens)
result = ROVER(tokenizer, detokenizer).fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.texts.rover.ROVER.fit.md)| Fits the model. The aggregated results are saved to the `texts_` attribute.
[fit_predict](crowdkit.aggregation.texts.rover.ROVER.fit_predict.md)| Fit the model and return the aggregated texts.
