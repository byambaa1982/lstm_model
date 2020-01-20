# LSTM Model

### What is LSTM?

LSTM stands for Long short-term-memory, meaning the short-term-memory is maintained in the LSTM cell state over long time steps. LSTM achieves this by overcoming the vanishing gradient problem that is typical of simpleRNN architecture.
Fork the project on github and git clone your fork, e.g.:

    git clone https://github.com/<username>/lstm_model.git
This is our data:
![Train test split](/images/data_pic.png)


The function below returns the above described windows of time for the model to train on. The parameter history_size is the size of the past window of information. The target_size is how far in the future does the model need to learn to predict. The target_size is the label that needs to be predicted.


    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

      start_index = start_index + history_size
      if end_index is None:
        end_index = len(dataset) - target_size

      for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
      return np.array(data), np.array(labels)
### Part 1: Forecast a univariate time series
First, you will train a model using only a single feature (Build Date), and use it to make predictions for that value in the future.

Let's first extract only the 'buildDate' from the dataset.

    uni_data = df['trend2min']
    uni_data.index = df['buildDate']
    uni_data.head() 

Let's observe how this data looks across time. 

    uni_data.plot(subplots=True)
    

![Train test split](/images/time_ser.png)
    



www.fiverr.com/coderjs
    

