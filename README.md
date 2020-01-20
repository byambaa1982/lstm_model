# LSTM Model

ALl code is here:

https://github.com/byambaa1982/lstm_model/blob/master/brunobuger.ipynb

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


	uni_data = uni_data.values

It is important to scale features before training a neural network. Standardization is a common way of doing this scaling by subtracting the mean and dividing by the standard deviation of each feature.You could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].

	
	uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
	uni_train_std = uni_data[:TRAIN_SPLIT].std()
    
Let's standardize the data.

	uni_data = (uni_data-uni_train_mean)/uni_train_std

Let's now create the data for the univariate model. For part 1, the model will be given the last 20 recorded temperature observations, and needs to learn to predict the temperature at the next time step.
	
	univariate_past_history = 20
	univariate_future_target = 0

	x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
	                                           univariate_past_history,
	                                           univariate_future_target)
	x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)


This is what the univariate_data function returns.


	print ('Single window of past history')
	print (x_train_uni[0])
	print ('\n Target temperature to predict')
	print (y_train_uni[0])
  


	Single window of past history
	[[ 0.56986556]
	 [-0.07101998]
	 [ 0.27407224]
	 [ 0.22477335]
	 [ 0.96425667]
	 [ 0.66846334]
	 [ 0.66846334]
	 [-0.41611219]
	 [ 0.47126779]
	 [ 0.4219689 ]
	 [ 0.47126779]
	 [-0.41611219]
	 [-0.16961775]
	 [ 0.91495778]
	 [ 0.37267001]
	 [-1.35279106]
	 [ 0.12617557]
	 [ 0.4219689 ]
	 [-0.31751442]
	 [-0.21891664]]

	 Target to predict
	1.0628544436459573


www.fiverr.com/coderjs
    

