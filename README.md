![Python application](https://github.com/joaofracasso/banknoteBrazil/workflows/Python%20application/badge.svg)
# e2easr: End to end speech recognition
This repository contains a end to end Brazilian speech recognition.

## Training Brazilian speech recognition

### Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/joaofracasso/e2e-asr-pt-br/blob/master/app/requirements.txt) dependencies installed. To install run:

```bash
$ python -m pip install -r app/requirements.txt
```

### Environments

Brazilian speech recognition may be run in any of the following up-to-date verified environments ([Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Codespace** See [Codespace](https://github.com/features/codespaces)
- **VS Code** See [Vs Code](https://code.visualstudio.com/docs/remote/containers)

### Train the network

You can train the network with the `train_model.py` script. For more usage information see [this](train_model.py). To train with the default parameters:

```bash
$ python app/train_model.py
```

### Evaluating the model

Also, you can evaluate the model against the validation set

```bash
$ python app/evaluate_model.py
```

## Predicting the outputs

To predict the outputs of a trained model using some dataset:

```bash
$  python app/predict_model.py --file data/test/2reaisVerso/compressed_20_9551306.jpeg
```

## Deploy on lambda container

Build the app Dockerfile:

```bash
docker build --pull --rm -f "app/Dockerfile" -t e2easr:latest "app" 
```

Run the app of bankNote:

```bash
docker run -p 8080:8080 e2easr:latest
```

Send send an (base64) audio over a POST request:

```bash
curl --location --request POST 'http://localhost:8080/2015-03-31/functions/function/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body": "audio/wav;base64"  
}' 
```

## Maintenance

This project is currently maintained by João Victor Calvo Fracasso and is for academic research use only. If you have any questions, feel free to contact joao.fracasso@outlook.com.

## License

The codes in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.