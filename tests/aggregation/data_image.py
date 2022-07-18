from typing import Tuple

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def simple_image_df() -> pd.DataFrame:
    im1_seg1 = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool)

    im1_seg2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool)

    im1_seg3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ], dtype=bool)

    im2_seg1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool)

    im2_seg2 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=bool)

    im2_seg3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ], dtype=bool)

    return pd.DataFrame(
        {
            'task': [1, 1, 1, 2, 2, 2],
            'worker': [
                'bf3a144d756790eb511f5ebccfcf3964',
                'be37db5784b50d08d2702f36317a3074',
                'e044b0849dfa9ce3dee5debbefb3b5da',
                'bf3a144d756790eb511f5ebccfcf3964',
                'be37db5784b50d08d2702f36317a3074',
                'e044b0849dfa9ce3dee5debbefb3b5da',
            ],
            'segmentation': [
                im1_seg1,
                im1_seg2,
                im1_seg3,
                im2_seg1,
                im2_seg2,
                im2_seg3,
            ],
            'image': [np.zeros((3, 8, 3)) for _ in range(6)],
        }
    )


@pytest.fixture
def image_with_skills_df() -> Tuple[pd.DataFrame, pd.Series]:

    im1_seg1 = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ], dtype=bool)

    im1_seg2 = np.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0]
    ], dtype=bool)

    im1_seg3 = np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=bool)

    return (
        pd.DataFrame(
            {
                'task': [1, 1, 1],
                'worker': [
                    'bf3a144d756790eb511f5ebccfcf3964',
                    'be37db5784b50d08d2702f36317a3074',
                    'e044b0849dfa9ce3dee5debbefb3b5da',
                ],
                'segmentation': [im1_seg1, im1_seg2, im1_seg3],
                'image': [np.zeros((3, 8, 3)) for _ in range(3)],
            }
        ),
        pd.Series(
            [1, 1, 3],
            pd.Index(
                [
                    'bf3a144d756790eb511f5ebccfcf3964',
                    'be37db5784b50d08d2702f36317a3074',
                    'e044b0849dfa9ce3dee5debbefb3b5da',
                ],
                name='worker',
            ),
        ),
    )


@pytest.fixture
def simple_image_mv_result() -> pd.Series:
    return pd.Series(
        [
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=bool),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ], dtype=bool),
        ],
        index=pd.Index([1, 2], name='task'),
        name='agg_segmentation'
    )


@pytest.fixture
def image_with_skills_mv_result() -> pd.Series:
    return pd.Series(
        [np.array([[0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)],
        index=pd.Index([1], name='task'),
        name='agg_segmentation'
    )


@pytest.fixture
def simple_image_rasa_result() -> pd.Series:
    return pd.Series(
        [
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=bool),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ], dtype=bool),
        ],
        index=pd.Index([1, 2], name='task'),
        name='agg_segmentation'
    )


@pytest.fixture
def simple_image_em_result() -> pd.Series:
    return pd.Series(
        [
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=bool),
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ], dtype=bool),
        ],
        index=pd.Index([1, 2], name='task'),
        name='agg_segmentation'
    )
