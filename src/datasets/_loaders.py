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
    data_name = 'relevance-2'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-2.zip'
    data_md5 = 'a39c3c30d9e946eeb80ca39954c96e95'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_csv(join(data_path, 'crowd_labels.csv'))
        true_labels = pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['label']

        return labels, true_labels

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)

    return load_dataframes(full_data_path)


def load_relevance5(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'relevance-5'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-5.zip'
    data_md5 = '4520f973003c7e151051e888edcd1a03'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_csv(join(data_path, 'crowd_labels.csv'))
        true_labels = pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['label'].rename('true_label')

        return labels, true_labels

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)

    return load_dataframes(full_data_path)


def load_mscoco(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'mscoco'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/mscoco.zip'
    data_md5 = 'e77224751f2ac1cb385c501c1ddd7393'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_pickle(join(data_path, 'crowd_labels.zip'))
        true_labels = pd.read_pickle(join(data_path, 'gt.pickle')).set_index('task')['true_segmentation']

        return labels, true_labels

    full_data_path = join(_load_dataset(data_name, get_data_dir(data_dir), data_url, data_md5), 'mscoco')

    return load_dataframes(full_data_path)


def load_mscoco_small(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'mscoco_small'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/mscoco_small.zip'
    data_md5 = '3705b8d3e96cab54af9a79cc35e354ab'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_pickle(join(data_path, 'crowd_labels.zip'))
        true_labels = pd.read_pickle(join(data_path, 'gt.zip'))

        return labels, true_labels

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)

    return load_dataframes(full_data_path)


def load_crowdspeech_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    labels = pd.read_csv(join(data_path, 'crowd_labels.csv')).rename(columns={'output': 'text'})
    true_labels = pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['output'].rename('true_label')

    return labels, true_labels


def load_crowdspeech_dev_clean(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'crowdspeech-dev-clean'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/crowdspeech-dev-clean.zip'
    data_md5 = 'f1cfbabcc9b93fd3beb14a9bdc3c4011'

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)
    return load_crowdspeech_dataframes(full_data_path)


def load_crowdspeech_dev_other(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'crowdspeech-dev-other'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/crowdspeech-dev-other.zip'
    data_md5 = 'ceb3e45c0f73f9446be43bd5ce4f646a'

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)
    return load_crowdspeech_dataframes(full_data_path)


def load_crowdspeech_test_clean(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'crowdspeech-test-clean'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/crowdspeech-test-clean.zip'
    data_md5 = 'd5db90c74c7b91a341d4669549b64ca9'

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)
    return load_crowdspeech_dataframes(full_data_path)


def load_crowdspeech_test_other(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'crowdspeech-test-other'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/crowdspeech-test-other.zip'
    data_md5 = '256fd3940ca6e86f883a2d37f7a23f3a'

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)
    return load_crowdspeech_dataframes(full_data_path)


def load_imdb_wiki_sbs(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'imdb_wiki'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/imdb-wiki-sbs.zip'
    data_md5 = '750f22901743492b63262931449818cc'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_csv(join(data_path, 'crowd_labels.csv'))
        labels.loc[labels['label'] == 'left', 'label'] = labels['left'].copy()
        labels.loc[labels['label'] == 'right', 'label'] = labels['right'].copy()

        true_labels = pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['score'].rename('true_label')

        return labels, true_labels

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)

    return load_dataframes(full_data_path)


def load_nist_trec_relevance(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data_name = 'nist-trec-relevance'
    data_url = 'https://tlk.s3.yandex.net/dataset/crowd-kit/relevance.zip'
    data_md5 = 'afc308986bb0cac865bf2434eb49a3f4'

    def load_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        labels = pd.read_csv(join(data_path, 'crowd_labels.csv'))
        true_labels = pd.read_csv(join(data_path, 'gt.csv')).set_index('task')['label']

        return labels, true_labels

    full_data_path = _load_dataset(data_name, data_dir, data_url, data_md5)

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
    'mscoco': {
        'loader': load_mscoco,
        'description': 'A sample of 2,000 images segmentations from MSCOCO dataset (https://cocodataset.org, licensed '
        'under Creative Commons Attribution 4.0 International Public License.) annotated on Toloka by 911 peformers. '
        'For each image, 9 workers submitted segmentations.'
    },
    'mscoco_small': {
        'loader': load_mscoco_small,
        'description': 'A sample of 100 images segmentations from MSCOCO dataset (https://cocodataset.org, licensed '
        'under Creative Commons Attribution 4.0 International Public License.) annotated on Toloka by 96 performers. '
        'For each image, 9 workers submitted segmentations.'
    },
    'crowdspeech-dev-clean': {
        'loader': load_crowdspeech_dev_clean,
        'description': 'This dataset is a publicly available large-scale dataset of crowdsourced audio transcriptions. '
        'It contains annotations for more than 20 hours of English speech from more than 1,000 crowd workers. '
        'This dataset corresponds to LibriSpeech (https://www.openslr.org/12,  licensed under CC BY 4.0) '
        'dev-clean dataset'
    },
    'crowdspeech-test-clean': {
        'loader': load_crowdspeech_test_clean,
        'description': 'This dataset is a publicly available large-scale dataset of crowdsourced audio transcriptions. '
        'It contains annotations for more than 20 hours of English speech from more than 1,000 crowd workers. '
        'This dataset corresponds to LibriSpeech (https://www.openslr.org/12, licensed under CC BY 4.0) '
        'test-clean dataset'
    },
    'crowdspeech-dev-other': {
        'loader': load_crowdspeech_dev_other,
        'description': 'This dataset is a publicly available large-scale dataset of crowdsourced audio transcriptions. '
        'It contains annotations for more than 20 hours of English speech from more than 1,000 crowd workers. '
        'This dataset corresponds to LibriSpeech (https://www.openslr.org/12, licensed under CC BY 4.0) '
        'dev-clean dataset'
    },
    'crowdspeech-test-other': {
        'loader': load_crowdspeech_test_other,
        'description': 'This dataset is a publicly available large-scale dataset of crowdsourced audio transcriptions. '
        'It contains annotations for more than 20 hours of English speech from more than 1,000 crowd workers. '
        'This dataset corresponds to LibriSpeech (https://www.openslr.org/12, licensed under CC BY 4.0) '
        'test-clean dataset'
    },
    'imdb-wiki-sbs': {
        'loader': load_imdb_wiki_sbs,
        'description': 'A sample of 2,497 images from the IMDB-WIKI dataset '
        '(https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) annotated on Toloka. This dataset contains images of '
        'people with reliable ground-truth age assigned to every image. The annotation allowed us to obtain 84,543 '
        'comparisons by 2,085 workers.'
    },
    'nist-trec-relevance': {
        'loader': load_nist_trec_relevance,
        'description': 'A dataset of English Web pages from the ClueWeb09 for English search queries which relevance '
        'were judged using crowdsourcing. This dataset was collected during NIST TREC Relevance Feedback Track 2010 '
        '(C.Buckley, M.Lease, and M.D.Smucker. Overview of the trec 2010 relevance feedback track (notebook). In The '
        'Nineteenth TREC Notebook, 2010.). There are 20,232 total (topic, document) examples (noisily) judged by 766 '
        'workers, who produced a total of 98,453 judgments. 3277 of the examples have prior "gold" labels by NIST.'
    }
}
