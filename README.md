# keysniff

typed text reconstruction based on audio only

## todo

* [x] data collection
* [x] data preprocessing
* [x] supervised learning
* [ ] deployment

## data preprocessing

* sync audio chunks with keypress
* create mel spectrogram
* convert keypress to model targets
* store results

## scripts

| name | function |
| ---: | :------- |
| analytics.py | Analyze collected data distribution |
| collect.py | Data collection pipeline |
| models.py | Pytorch deep learning models |
| preprocess.py | Data preprocessing, creating single data file |
| test.py | Testing validation results |
| train.py | Used to train models from models.py |
| util.py | Utility functions |