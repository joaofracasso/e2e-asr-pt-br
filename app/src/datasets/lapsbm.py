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

URL = "lapsbm-test"
FOLDER_IN_ARCHIVE = "lapsbm"
_CHECKSUMS = {
    "http://www02.smt.ufrj.br/~igor.quintanilha/lapsbm-val.tar.gz":
    "9df986eb828545bb99d2ed89080951fdb24daad80f6e2146c0b5b5150544f99a",
    "http://www02.smt.ufrj.br/~igor.quintanilha/lapsbm-test.tar.gz":
    "200fc88e15eb0a442c1b5be849a83900d5868f2d958bd5b82b2ad0c419cf23a1"
}


def load_lapsbm_item(fileid: str,
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


class Lapsbm(Dataset):
    """Create a Dataset for lapsbm.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"lapsbm"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
                "lapsbm-val",
                "lapsbm-test",
        ]:
            ext_archive = ".tar.gz"
            base_url = "http://www02.smt.ufrj.br/~igor.quintanilha/"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(URL, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive, self._path)

        self._walker = sorted(str(p) for p in Path(self._path).glob('*/*' + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return load_lapsbm_item(fileid, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
