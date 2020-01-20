# LSTM Model

### What is LSTM?

## Setup
LSTM stands for Long short-term-memory, meaning the short-term-memory is maintained in the LSTM cell state over long time steps. LSTM achieves this by overcoming the vanishing gradient problem that is typical of simpleRNN architecture.
Fork the project on github and git clone your fork, e.g.:

    git clone https://github.com/<username>/lstm_model.git

Create a virtualenv using Python 3 and install dependencies. I recommend getting python3 using a package manager (homebrew on OSX), then installing [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation) to that python. NOTE! You must change 'path/to/python3'
to be the actual path to python3 on your system.

    mkvirtualenv flask-google-sheets --python=/path/to/python3
    pip install -r requirements.txt

