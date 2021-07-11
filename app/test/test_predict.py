import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
import onnxruntime as ort
import pytest
from predict_model import get_prediction
from src.datasets.lapsbm import Lapsbm


if not os.path.isdir("./data"):
    os.makedirs("./data")

test_dataset = Lapsbm("./data", url="lapsbm-test", download=True)


@pytest.mark.parametrize('audio', test_dataset._walker)
def test_get_prediction(audio):
    ort_session = ort.InferenceSession('app/models/e2e_asr_best.onnx')
    with open(audio, 'rb') as f:
        audio_bytes = f.read()
        class_ = get_prediction(audio_bytes, ort_session)
    assert isinstance(class_, str)
