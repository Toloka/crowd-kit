# TextSummarization
`crowdkit.aggregation.texts.text_summarization.TextSummarization` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/text_summarization.py#L15)

```python
TextSummarization(
    self,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    concat_token: str = ' | ',
    num_beams: int = 16,
    n_permutations: Optional[int] = None,
    permutation_aggregator: Optional[BaseTextsAggregator] = None,
    device: str = 'cpu'
)
```

Text Aggregation through Summarization


The method uses a pre-trained language model for summarization to aggregate crowdsourced texts.
For each task, texts are concateneted by ` | ` token and passed as a model's input. If
`n_permutations` is not `None`, texts are random shuffled `n_permutations` times and then
outputs are aggregated with `permutation_aggregator` if provided. If `permutation_aggregator`
is not provided, the resulting aggregate is the most common output over permuted inputs.

**To use pretrained model and tokenizer from `transformers`, you need to install [torch](https://pytorch.org/get-started/locally/#start-locally)**

M. Orzhenovskii,
"Fine-Tuning Pre-Trained Language Model for Crowdsourced Texts Aggregation,"
Proceedings of the 2nd Crowd Science Workshop: Trust, Ethics, and Excellence in Crowdsourced Data Management at Scale, 2021, pp. 8-14.
<https://ceur-ws.org/Vol-2932/short1.pdf>

S. Pletenev,
"Noisy Text Sequences Aggregation as a Summarization Subtask,"
Proceedings of the 2nd Crowd Science Workshop: Trust, Ethics, and Excellence in Crowdsourced Data Management at Scale, 2021, pp. 15-20.
<https://ceur-ws.org/Vol-2932/short2.pdf>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`tokenizer`|**PreTrainedTokenizer**|<p>[Pre-trained tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizer).</p>
`model`|**PreTrainedModel**|<p>[Pre-trained model](https://huggingface.co/transformers/main_classes/model.html#pretrainedmodel) for text summarization.</p>
`concat_token`|**str**|<p>Token used for the workers&#x27; texts concatenation. </p><p>Default value: ` | `.</p>
`num_beams`|**int**|<p>Number of beams for beam search. 1 means no beam search. </p><p>Default value: `16`.</p>
`n_permutations`|**Optional\[int\]**|<p>Number of input permutations to use. If `None`, use a single permutation according to the input&#x27;s order. </p><p>Default value: `None`.</p>
`permutation_aggregator`|**Optional\[[BaseTextsAggregator](crowdkit.aggregation.base.BaseTextsAggregator.md)\]**|<p>Text aggregation method to use for aggregating outputs of multiple input permutations if `use_permutations` flag is set. </p><p>Default value: `None`.</p>
`device`|**str**|<p>Device to use such as `cpu` or `cuda`. </p><p>Default value: `cpu`.</p>
`texts_`|**Series**|<p>Tasks&#x27; texts. A pandas.Series indexed by `task` such that `result.loc[task, text]` is the task&#x27;s text.</p>

**Examples:**

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from crowdkit.aggregation import TextSummarization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mname = "toloka/t5-large-for-text-aggregation"
tokenizer = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSeq2SeqLM.from_pretrained(mname)
agg = TextSummarization(tokenizer, model, device=device)
result = agg.fit_predict(df)
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit_predict](crowdkit.aggregation.texts.text_summarization.TextSummarization.fit_predict.md)| Run the aggregation and return the aggregated texts.
