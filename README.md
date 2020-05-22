# Character Level Twitter Sentiment Analysis

## Description
Implements a straightforward character level LSTM network for Twitter sentiment analysis. There is a `/utils/read_dataset` class which reads in the data and sets parameters but is primarily leaning on https://charlesashby.github.io/2017/06/05/sentiment-analysis-with-char-lstm/

## Use
Clone the repository, activate your virtual environment and enter:
`cd CharLSTM && pip install -r requirements`.

Use the `test_notebook.ipynb` as an example of how the functions are expected to be run. 

The code base relies on the `tokenize` package so you will need to perform a once off installation of the `nltk` module. Simply uncomment `nltk.download()` in the `test_notebook` and run the first cell. A GUI will show up - select d for Download and enter 'punkt'. This should lead to the correct install and then you're done with setting up :)
