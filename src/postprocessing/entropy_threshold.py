import numpy as np
import warnings

from ..aggregation import annotations
from ..metrics.data import uncertainty


def entropy_threshold(
    answers: annotations.LABELED_DATA,
    performers_skills: annotations.OPTIONAL_SKILLS = None,
    percentile: int = 10,
    min_answers: int = 2,
):
    """
    Entropy thresholding postprocessing: filters out all answers by performers,
    whos' entropy (uncertanity) of answers is below specified percentile. This heuristic detects answers of performers
    that answer the same way too often, e.g. when "speed-running" by only clicking one button.
    Args:
        answers (pandas.DataFrame): A data frame containing `task`, `performer` and `label` columns.
        performers_skills (Optional[pandas.Series]): performers skills e.g. golden set skills.
        percentile (int): threshold entropy percentile from 0 to 100. Default: 10.
        min_answers (int): performer can be filtered out if he left at least that many answers.

    Returns:
        pd.DataFrame

    Examples:
        Fraudent performer always answers the same and gets filtered out.

        >>>answers = pd.DataFrame.from_records(
        >>>     [
        >>>         {'task': '1', 'performer': 'A', 'label': frozenset(['dog'])},
        >>>         {'task': '1', 'performer': 'B', 'label': frozenset(['cat'])},
        >>>         {'task': '2', 'performer': 'A', 'label': frozenset(['cat'])},
        >>>         {'task': '2', 'performer': 'B', 'label': frozenset(['cat'])},
        >>>         {'task': '3', 'performer': 'A', 'label': frozenset(['dog'])},
        >>>         {'task': '3', 'performer': 'B', 'label': frozenset(['cat'])},
        >>>     ]
        >>>)
        >>>entropy_threshold(answers)
          task performer  label
        0    1         A  (dog)
        2    2         A  (cat)
        4    3         A  (dog)
    """

    answers_per_performer = answers.groupby('performer')['label'].count()
    answers_per_performer = answers_per_performer[answers_per_performer >= min_answers]
    answers_for_filtration = answers[answers.performer.isin(answers_per_performer.index)]

    uncertainties = uncertainty(
        answers_for_filtration,
        performers_skills,
        compute_by='performer',
        aggregate=False,
    )
    cutoff = np.percentile(uncertainties, percentile)
    removed_performers = uncertainties[uncertainties <= cutoff].index
    filtered_answers = answers.copy(deep=False)
    filtered_answers = filtered_answers[~filtered_answers['performer'].isin(removed_performers)]
    if filtered_answers.shape[0] <= answers.shape[0]//2:
        warnings.warn('Removed >= 1/2 of answers with entropy_threshold. This might lead to poor annotation quality. '
                      'Try decreasing percentile or min_answers.')
    return filtered_answers
