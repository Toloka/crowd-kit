__all__ = [
    'load_dataset',
    'get_datasets_list',
]
import pandas
import typing


def load_dataset(dataset: str, data_dir: typing.Optional[str] = None) -> typing.Tuple[pandas.DataFrame, pandas.Series]:
    """Downloads a dataset from remote and loads it into Pandas objects.
    If a dataset is already downloaded, loads it from cache.

    Parameters:
        dataset: str, a dataset name
        data_dir: Optional[str]
            Path to folder where to store downloaded dataset. If `None`, `~/crowdkit_data` is used.
            `default=None`. Alternatively, it can be set by the 'CROWDKIT_DATA' environment variable.
    Returns:
        data: Tuple[pd.DataFrame, pd.Series], a tuple of workers answers and ground truth labels.
    """
    ...


def get_datasets_list() -> typing.List[typing.Tuple[str, str]]:
    """Returns a list of available datasets in format [(name, description)].
    """
    ...
