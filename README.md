In this study, we developed a Long Short-Term Memory model to predict the onset of sepsis at different time intervals (2 hours, 4 hours, and 6 hours in advance). The model consists of three LSTM layers with 128 neurons a dropout rate of 0.2 applied between layers and after the output layer, and three fully connected layer with one units each. The training was done in four folds over a batch size of 512 samples, a learning rate of 1e-5, weight decay of 1e-4. The goal was to assess the model's performance, analyze the error scores, and evaluate its advantages and limitations for prediction tasks.

### Model architecture and hyperparameters
The LSTM model was selected due to its strength in handling sequential data, particularly time-series data like those used in predicting sepsis onset. The three-layer LSTM structure, combined with 128 neurons per layer, provides a balance between model complexity and capacity for learning temporal dependencies. The choice of dropout at 0.2 helps prevent overfitting by introducing regularization.
The use of the sigmoid activation function for the output layer is appropriate, as it maps the model’s predictions to a probability between 0 and 1, aligning with the binary classification nature of the task, sepsis onset or no sepsis.

### Training results 
The model was trained over multiple folds, with the loss consistently decreasing across epochs, as evidenced by the loss and validation loss values during training. However, the validation loss did not consistently decrease in tandem with the training loss, indicating some potential overfitting or issues with generalization.

The observed fluctuations in the validation loss, with slight increases at certain points, suggest that the model might struggle with some aspects of the data, potentially due to noise. These fluctuations are typical in complex models like LSTMs and may reflect the model’s difficulty in generalizing to unseen data.

### Error score and Gradient-based feature importance
The model’s performance can be assessed using appropriate evaluation metrics like accuracy and F1-Score. 
We have determined, for each fold, the evaluation metrics of each prediction. Moreover, we have averaged the results for each predicted target. Cross-validation use different training and testing sets, so the model may perform slightly differently depending on the data. Averaging the scores from each fold helps smooth out fluctuations and gives a more reliable estimate of performance. It can also help to mitigate bias, a model may perform unusually well or poorly on a particular fold due to the way the data is distributed. By averaging across multiple folds, we reduce the impact of such biases.

The highest overall accuracy was obtained for the two hours in advance sepsis prediction target with a score of 70,58% and a F1-score of 69,55%. The worst results are those of the 6 hours in advance prediction with scores of 70,49% and F1-score of 69,70%. We can see that the model predicts in a better way events in the near future. Nevertheless, this results are quite close. We can espect similar outcomes.

As for the feature importance, we have see that for every prediction target, we have the same top 5 most influencial features.  In this case, the gradient values of the features "lactate","c-reactive_protein", "bilirubin", "age" and the "respiratory_minute_volume" significantly impact the model's predictions.

### Limitations
We have encountered several limitations during this project. 
1. Overfitting: Despite the dropout, the model still experiences fluctuations in validation loss, suggesting that it may be overfitting to the training data. This could be due to an insufficient amount of training data or overly complex model architecture for the dataset.
2. Computational Cost: These LSTM models, especially with multiple layers and large batch sizes, are computationally expensive. This might be a limitation in real-time clinical applications, where prediction speed and resource constraints are critical.    
