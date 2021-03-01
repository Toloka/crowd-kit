from typing import Any, Callable, Optional, Union
import pandas as pd


def golden_set_accuracy(answers: pd.DataFrame,
                        golden_answers: pd.Series,
                        by_performer: bool = False,
                        answer_column: Any = 'label',
                        comapre_function: Optional[Callable[[Any, Any], float]] = None) -> Union[float, pd.Series]:
    """
    Golden set accuracy metric: a fraction of worker's correct answers on golden tasks.
    Args:
            answers (pandas.DataFrame): A data frame containing `task`, `performer` and `label` columns.
            golden_answers (pandas.Series): ground-truth answers for golden tasks.
            by_performer (bool): if set, returns accuracies for every performer in provided data frame. Otherwise,
                returns an average accuracy of all performers.
            answer_column: column in the data frame that contanes performers answers.
            comapre_function (Optional[Callable[[Any, Any], float]]): function that compares performer's answer with
                the golden answer. If `None`, uses `==` operator.
        Returns:
            Union[float, pd.Series]
    """
    answers = answers.copy(deep=False)
    answers.set_index('task', inplace=True)
    answers['golden'] = golden_answers
    answers = answers[answers.golden.notna()]
    if comapre_function is None:
        answers['skill'] = answers[answer_column] == answers['golden']
    else:
        answers['skill'] = answers.apply(lambda row: comapre_function(row[answer_column], row['golden']), axis=1)

    if by_performer:
        performers_skills = answers.groupby('performer').sum('skill')['skill']
        return performers_skills / answers.groupby('performer').count()['label']
    else:
        return answers['skill'].mean()
