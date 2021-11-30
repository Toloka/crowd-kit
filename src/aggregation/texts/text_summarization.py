__all__ = [
    'TextSummarization'
]

import attr
import itertools
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Optional

from ..base import BaseTextsAggregator
from ..annotations import TEXT_DATA, TASKS_TEXTS, manage_docstring


@attr.s
@manage_docstring
class TextSummarization(BaseTextsAggregator):
    """Text Aggregation through Summarization

    The method uses a pre-trained language model for summarization to aggregate crowdsourced texts.
    For each task, texts are concateneted by ` | ` token and passed as a model's input. If
    `n_permutations` is not `None`, texts are random shuffled `n_permutations` times and then
    outputs are aggregated with `permutation_aggregator` if provided. If `permutation_aggregator`
    is not provided, the resulting aggregate is the most common output over permuted inputs.

    **To use pretrained model and tokenizer from `transformers`, you need to install [torch](https://pytorch.org/get-started/locally/#start-locally)**

    M. Orzhenovskii,
    "Fine-Tuning Pre-Trained Language Model for Crowdsourced Texts Aggregation,"
    Proceedings of the 2nd Crowd Science Workshop: Trust, Ethics, and Excellence in Crowdsourced Data Management at Scale, 2021, pp. 8-14.
    http://ceur-ws.org/Vol-2932/short1.pdf

    S. Pletenev,
    "Noisy Text Sequences Aggregation as a Summarization Subtask,"
    Proceedings of the 2nd Crowd Science Workshop: Trust, Ethics, and Excellence in Crowdsourced Data Management at Scale, 2021, pp. 15-20.
    http://ceur-ws.org/Vol-2932/short2.pdf

    Args:
        tokenizer: [Pre-trained tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizer).
        model: [Pre-trained model](https://huggingface.co/transformers/main_classes/model.html#pretrainedmodel) for text summarization.
        concat_token: Token used for the performers' texts concatenation.
            Default value: ` | `.
        num_beams: Number of beams for beam search. 1 means no beam search.
            Default value: `16`.
        n_permutations: Number of input permutations to use. If `None`, use a single permutation according to the input's order.
            Default value: `None`.
        permutation_aggregator: Text aggregation method to use for aggregating outputs of multiple input permutations if `use_permutations` flag is set.
            Default value: `None`.
        device: Device to use such as `cpu` or `cuda`.
            Default value: `cpu`.
    Example:
        >>> import torch
        >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
        >>> from crowdkit.aggregation import TextSummarization
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> mname = "toloka/t5-large-for-text-aggregation"
        >>> tokenizer = AutoTokenizer.from_pretrained(mname)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained(mname)
        >>> agg = TextSummarization(tokenizer, model, device=device)
        >>> result = agg.fit_predict(df)
        ...
    """
    tokenizer: PreTrainedTokenizer = attr.ib()
    model: PreTrainedModel = attr.ib()
    concat_token: str = attr.ib(default=' | ')
    num_beams: int = attr.ib(default=16)
    n_permutations: Optional[int] = attr.ib(default=None)
    permutation_aggregator: Optional[BaseTextsAggregator] = attr.ib(default=None)
    device: str = attr.ib(default='cpu')
    # texts_

    @manage_docstring
    def fit_predict(self, data: TEXT_DATA) -> TASKS_TEXTS:
        data = data[['task', 'performer', 'text']]

        self.model = self.model.to(self.device)
        self.texts_ = data.groupby('task')['text'].apply(self._aggregate_one)
        return self.texts_

    @manage_docstring
    def _aggregate_one(self, outputs: TASKS_TEXTS) -> str:
        if not self.n_permutations:
            return self._generate_output(outputs)

        generated_outputs = []

        # TODO: generate only `n_permutations` permutations
        permutations = list(itertools.permutations(outputs))
        permutations_idx = np.random.choice(len(permutations), size=self.n_permutations, replace=False)
        permutations = [permutations[i] for i in permutations_idx]
        for permutation in permutations:
            generated_outputs.append(self._generate_output(permutation))

        data = pd.DataFrame({'task': [''] * len(generated_outputs), 'text': generated_outputs})
        if self.permutation_aggregator is not None:
            return self.permutation_aggregator.fit_predict(data)['']
        return data.text.mode()

    @manage_docstring
    def _generate_output(self, permutation: TASKS_TEXTS) -> str:
        input_text = self.concat_token.join(permutation)
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(input_ids, num_beams=self.num_beams)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
