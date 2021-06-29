import base64
import json
import io

import torchaudio
import onnxruntime as ort

from predict_model import get_prediction

ort_session = ort.InferenceSession('models/e2e_asr_best.onnx')

#lambda invocation
def lambda_handler(event, context):
    audio_bytes = base64.b64decode(event["body"])
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    print(waveform)
    output = get_prediction(audio_bytes, ort_session)
    print(output)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'trans': output
        })
    }
