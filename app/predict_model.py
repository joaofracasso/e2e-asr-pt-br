import io
import argparse

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from torchaudio import transforms
from torchaudio.transforms import MelSpectrogram
from src.build_features import text_transform

def GreedyDecoderPred(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
        return decodes


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
        return decodes, targets


def get_prediction(audio_bytes, inference_session):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    audio_transform = MelSpectrogram(sample_rate=sample_rate)
    spectrograms = audio_transform(waveform).squeeze(0).transpose(0, 1)
    spectrograms = nn.utils.rnn.pad_sequence([spectrograms], batch_first=True).unsqueeze(1).transpose(2, 3)
    output = inference_session.run(None, {'input': spectrograms.numpy()})
    output = torch.tensor(output)[0]
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1) 
    return GreedyDecoderPred(output.transpose(0, 1))

if __name__ == "__main__":
    ort_session = ort.InferenceSession('app/models/e2e_asr_best.onnx')
    file_audio = [
        "data/lapsbm-val/LapsBM-F006/LapsBM_0103.wav"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='app/models/e2e_asr_best.onnx', type=str)
    parser.add_argument('--file', default=None, type=str)
    opt = parser.parse_args()
    with open(file_audio[0], 'rb') as f:
        audio_bytes = f.read()
    output = get_prediction(audio_bytes,ort_session)
    print(output)
