import os
from typing import Tuple, Union
from pathlib import Path
import re

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
from unidecode import unidecode

URL = "alcaim"
FOLDER_IN_ARCHIVE = "alcaim"
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz":
    "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
}


def load_alcaim_item(fileid: str,
                          ext_audio: str,
                          ext_txt: str) -> Tuple[Tensor, int, str, int, int, int]:
    
    file_text = os.path.splitext(fileid)[0] + ext_txt

    # Load audio
    waveform, sample_rate = torchaudio.load(fileid)

    # Load text
    with open(file_text, 'r', encoding='utf8') as ft:
        utterance = ft.readlines()[0].strip()

    return (
        waveform,
        sample_rate,
        re.sub('[^A-Za-z ]+', '', unidecode(utterance)),
    )


class Alcaim(Dataset):
    """Create a Dataset for alcaim.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"alcaim"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self,
                 root: Union[str, Path],
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:


        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        
        self._path = os.path.join(root, folder_in_archive)

        #if download:
        #    if not os.path.isdir(self._path):
        #        if not os.path.isfile(archive):
        #            checksum = _CHECKSUMS.get(URL, None)
        #            download_url(URL, root, hash_value=checksum)
        #        extract_archive(archive)

        self._walker = sorted(str(p) for p in Path(self._path).glob('*/*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_alcaim_item(fileid, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
