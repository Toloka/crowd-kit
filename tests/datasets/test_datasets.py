from crowdkit.datasets import get_datasets_list


def test_get_datasets_list():
    available_datasets = {'relevance-2', 'relevance-5'}
    datasets_list = {dataset for dataset, description in get_datasets_list()}
    assert len(available_datasets) == len(available_datasets & datasets_list)
