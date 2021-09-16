import pandas as pd

from ._base import get_data_dir, fetch_remote

from os.path import exists, join
from typing import Optional, Tuple


def _load_dataset(data_name, data_dir, data_url, data_md5):
    data_dir = get_data_dir(data_dir)
    full_data_path = join(data_dir, data_name)

    if not exists(full_data_path):
        print(f'Downloading {data_name} from remote')
        fetch_remote(data_url, data_md5, full_data_path + '.zip', data_dir)

    return full_data_path


def load_relevance2(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    DATA_NAME = 'relevance-2'
    DATA_URL = 'https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-2.zip'
    DATA_MD5 = 'a39c3c30d9e946eeb80ca39954c96e95'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        return pd.read_csv(join(data_path, 'crowd_labels.csv')), \
            pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['label']

    full_data_path = _load_dataset(DATA_NAME, data_dir, DATA_URL, DATA_MD5)

    return load_dataframes(full_data_path)


def load_relevance5(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    DATA_NAME = 'relevance-5'
    DATA_URL = 'https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-5.zip'
    DATA_MD5 = '4520f973003c7e151051e888edcd1a03'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        return pd.read_csv(join(data_path, 'crowd_labels.csv')), \
            pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['label']

    full_data_path = _load_dataset(DATA_NAME, data_dir, DATA_URL, DATA_MD5)

    return load_dataframes(full_data_path)


DATA_LOADERS = {
    'relevance-2': {
        'loader': load_relevance2,
        'description': 'This dataset, designed for evaluating answer aggregation methods in crowdsourcing, '
        'contains around 0.5 million anonymized crowdsourced labels collected in the Relevance 2 Gradations project'
        ' in 2016 at Yandex. In this project, query-document pairs are provided with binary labels: relevant or non-relevant.'
    },
    'relevance-5': {
        'loader': load_relevance5,
        'description': 'This dataset was designed for evaluating answer aggregation methods in crowdsourcing. '
        'It contains around 1 million anonymized crowdsourced labels collected in the Relevance 5 Gradations project'
        ' in 2016 at Yandex. In this project, query-document pairs are labeled on a scale of 1 to 5. from least relevant'
        ' to most relevant.'
    },
}
