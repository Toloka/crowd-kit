import pandas as pd
from os.path import exists, join
from typing import Optional, List, Tuple

from ._base import get_data_home, fetch_remote, load_dataframes
from ._data_urls import DATA_URLS


def load_dataset(dataset: str, data_home: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Downloads a dataset from remote and loads it into Pandas objects.
    If a dataset is already downloaded, loads it from cache.

    Parameters:
        dataset: str, a dataset name
        data_home: Optional[str], path to folder where to store downloaded dataset.
        If `None`, `~/crowdkit_data` is used. `default=None`
    Returns:
        data: Tuple[pd.DataFrame, pd.Series], a tuple of performers andswers and ground truth labels.
    """
    data_home = get_data_home(data_home)
    if exists(join(data_home, dataset)):
        return load_dataframes(join(data_home, dataset))

    if dataset not in DATA_URLS:
        raise ValueError('This dataset does not exist')
    print(f'Downloading {dataset} from remote')
    fetch_remote(DATA_URLS[dataset]['link'], DATA_URLS[dataset]['md5'], join(data_home, dataset) + '.zip', data_home)
    return load_dataframes(join(data_home, dataset))


def get_datasets_list() -> List[Tuple[str, str]]:
    """Returns a list of available datasets in format [(name, description)]."""
    return [(dataset, info['description']) for dataset, info in DATA_URLS.items()]
