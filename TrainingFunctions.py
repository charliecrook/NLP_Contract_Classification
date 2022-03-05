
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def train_data(X_train, y_train, X_val, y_val, model, loss, optimizer, epochs, batch_size, callbacks, model_version, model_name):
  """
  Parameters:
  X_train: (dataframe) training dataset of predictors/attributes
  y_train: (dataframe) training dataset of classification target
  X_val: (dataframe) validation dataset of predictors/attributes
  y_val: (dataframe) validation dataset of classification target
  model: (keras model) model to be trained
  loss: (str) loss function to be used during training
  optimiser: (keras optimiser) optimiser to be used during training
  epochs: (int) number of epochs
  callbacks: (keras callbacks) List of callbacks to apply during training.
  class_weight: (dic) Optional dictionary mapping class indices (integers) to a weight (float) value
  model_version: (str) String indicating the model version
  model_name: (str) String indicating the name of the current training instance
  
  Returns:
  history: A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values
  """
  
  model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  
  history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[callbacks], shuffle=True, verbose=1)
  
  plt.rcParams["figure.figsize"] = (8, 6)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train loss', 'val loss'])
  plt.grid(True)

  return history

def performance_report(X_test, y_test, model):
  """
  Parameters:
  X_test: (dataframe) test dataset of unseen predictors/attributes
  y_test: (dataframe) test dataset of unseen classification target
  model: (keras model) Trained model
  
  Returns:
  Prints a report of the classifier performance including accuracy
  score, f1 score and confusion matrix
  """
  y_pred = model.predict(X_test)
  y_pred = y_pred > 0.5
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')
  
  

  try:
    total_loss = model.evaluate(X_test,y_test)
    print('total loss:', total_loss)
  except:
    print('This must be a SVM model..')  
  finally:
    print('Global accuracy:', acc)
    print('Global f1 score:', f1)
    print('\n')
    print(multilabel_confusion_matrix(y_test, y_pred))

def kaggle_prediction(test_X, model, name):
  """
  Parameters:
  test: (dataframe) unseen test data
  model: (keras model) Trained model
  name: (str) output csv file name

  Returns:
  submission_file: (csv) submission file ready to be uploaded to kaggle
  """

  predictions = model.predict(test_X)
  predictions = predictions > 0.5
  results = []
  for prediction in predictions:
      results.append('')
      for p in prediction:
          results[-1] += '1' if p else '0'
  test_predictions = pd.DataFrame(results, columns=['label'])

  final_submission = result_ids.join(test_predictions)
  final_submission.reset_index(drop=True)
  final_submission.to_csv(name, index=False)