import os

import onnxruntime as ort
import torch.utils.data as data

from src.build_features import cer, data_processing, wer
from src.datasets.lapsbm import Lapsbm
from predict_model import get_transcription, GreedyDecoder

data_dir = "data"
batch_size = 1

if __name__ == "__main__":
    ort_session = ort.InferenceSession('app/models/e2e_asr_best.onnx')
    if not os.path.isdir("./data"):
        os.makedirs("./data")
    test_dataset = Lapsbm("./data", url="lapsbm-test", download=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'))
    test_cer, test_wer = [], []
    for i, _data in enumerate(test_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        output = get_transcription(spectrograms, ort_session)
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    print("Target :{}\nPredict :{}".format(decoded_targets[j], decoded_preds[j]))
    print('Test set: Average CER: {:4f} Average WER: {:.4f}\n'.format(avg_cer, avg_wer))
