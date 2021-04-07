from hashlib import md5
from os import environ, makedirs
from os.path import exists, expanduser, join
import pandas as pd
from shutil import unpack_archive
from typing import Optional, Tuple
from urllib.request import urlretrieve


def get_data_home(data_home: Optional[str] = None) -> str:
    """Return the path of the crowd-kit data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'crowdkit_data' in the
    user home folder.
    Alternatively, it can be set by the 'CROWDKIT_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Parameters:
        data_home: str, default=None
            The path to crowd-kit data directory. If `None`, the default path
            is `~/crowdkit_data`.
    """
    if data_home is None:
        data_home = environ.get('CROWDKIT_DATA', join('~', 'crowdkit_data'))
        data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def load_dataframes(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads dataset's csv files into Pandas data frames.

    Parameters:
        path: str, path dataset's directory with files
        `crowd_labels.csv` and `gt.csv`.
    """
    crowd_labels = pd.read_csv(join(path, 'crowd_labels.csv'))
    ground_truth = pd.read_csv(join(path, 'gt.csv'))
    return crowd_labels, ground_truth.set_index('task')['label']


def fetch_remote(url: str, checksum: str, path: str, data_home: str) -> None:
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the MD5 Checksum of the
    downloaded file.
    Parameters:
        url: str
        checksum: str
        path: str, path to save a zip file
        data_home: path to crowd-kit data directory
    """
    urlretrieve(url, path)
    fetched_checksum = md5(open(path, 'rb').read()).hexdigest()
    if checksum != fetched_checksum:
        raise IOError(f"{path} has an MD5 checksum ({fetched_checksum}) differing from expected ({checksum}), file may be corrupted.")
    unpack_archive(path, data_home)
