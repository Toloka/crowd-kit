from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from crowdkit.datasets import get_datasets_list, load_dataset


def test_get_datasets_list() -> None:
    available_datasets = {'relevance-2', 'relevance-5'}
    datasets_list = {dataset for dataset, description in get_datasets_list()}
    assert len(available_datasets) == len(available_datasets & datasets_list)


def collect_stats_for_dataset(crowd_labels: pd.DataFrame, gt: pd.Series) -> Dict[str, Any]:
    return {
        'rows': len(crowd_labels),
        'dtypes_labels': crowd_labels.dtypes.to_dict(),
        'dtypes_gt': gt.dtypes,
        'name_gt': gt.name,
        'index_name_gt': gt.index.name,
    }


@pytest.mark.parametrize(
    'dataset, dataset_stats',
    [
        ('relevance-2', {
            'rows': 475536,
            'dtypes_labels': {
                'worker': object,
                'task': object,
                'label': np.int64,
            },
            'dtypes_gt': np.int64,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('relevance-5', {
            'rows': 1091918,
            'dtypes_labels': {
                'worker': object,
                'task': object,
                'label': np.int64,
            },
            'dtypes_gt': np.int64,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('mscoco', {
            'rows': 18000,
            'dtypes_labels': {
                'worker': object,
                'task': np.int64,
                'segmentation': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_segmentation',
            'index_name_gt': 'task',
        }),
        ('mscoco_small', {
            'rows': 900,
            'dtypes_labels': {
                'worker': object,
                'task': np.int64,
                'segmentation': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_segmentation',
            'index_name_gt': 'task',
        }),
        ('crowdspeech-dev-clean', {
            'rows': 18921,
            'dtypes_labels': {
                'worker': np.int64,
                'task': object,
                'text': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('crowdspeech-test-clean', {
            'rows': 18340,
            'dtypes_labels': {
                'worker': np.int64,
                'task': object,
                'text': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('crowdspeech-dev-other', {
            'rows': 20048,
            'dtypes_labels': {
                'worker': np.int64,
                'task': object,
                'text': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('crowdspeech-test-other', {
            'rows': 20573,
            'dtypes_labels': {
                'worker': np.int64,
                'task': object,
                'text': object,
            },
            'dtypes_gt': object,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
        ('imdb-wiki-sbs',  {
            'rows': 250249,
            'dtypes_labels': {
                'worker': np.int64,
                'left': object,
                'right': object,
                'label': object,
            },
            'dtypes_gt': np.int64,
            'name_gt': 'true_label',
            'index_name_gt': 'label',
        }),
        ('nist-trec-relevance', {
            'rows': 98453,
            'dtypes_labels': {
                'worker': object,
                'task': object,
                'label': np.int64,
            },
            'dtypes_gt': np.int64,
            'name_gt': 'true_label',
            'index_name_gt': 'task',
        }),
    ]
)
def test_load_dataset(dataset: str, dataset_stats: Dict[str, Any], tmpdir_factory: Any) -> None:
    data_dir = tmpdir_factory.mktemp('crowdkit_data')
    df, gt = load_dataset(dataset, data_dir=data_dir)
    assert collect_stats_for_dataset(df, gt) == dataset_stats
