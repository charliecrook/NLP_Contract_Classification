
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self


class CombineColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
        self.original_columns = None


    def fit(self, X ,y=None):
        self.original_columns = X.columns
        return self


    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, self.original_columns)

        X['combined'] = X[self.columns].agg(' '.join, axis = 1)
        remaining_cols = X.drop(self.columns, axis=1)
        return remaining_cols


class Tokenization(BaseEstimator, TransformerMixin):
  def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=True)

  def fit(self, X ,y=None):
        return self

  def transform(self, X):
        tokens = list(map(lambda t: ['[CLS]'] + self.tokenizer.tokenize(t)[:510] + ['[SEP]'], X['combined']))
        tokens_ids = pad_sequences(list(map(self.tokenizer.convert_tokens_to_ids, tokens)), maxlen=512, truncating="post", padding="post", dtype="int")
        token = pd.DataFrame(tokens_ids, columns =[f'Title_{i}' for i in range(512)]) 
        X = X.join(token)
        X.drop('combined', axis=1, inplace=True)
        return X


class FillNaN(BaseEstimator, TransformerMixin):
  def __init__(self):
        super().__init__()

  def fit(self, X ,y=None):
        return self

  def transform(self, X):
        X = X.fillna(X['nature_of_contract'].value_counts().index[0])
        X['nature_of_contract'] = X['nature_of_contract'].replace('combined', 'works')
        return X


class OneHot(BaseEstimator, TransformerMixin):
  def __init__(self):
        super().__init__()
        self.encoder_contract_type = OneHotEncoder(drop='first', sparse=False)
        self.encoder_nature_of_contract = OneHotEncoder(drop='first', sparse=False)

  def fit(self, X ,y=None):
        self.encoder_contract_type.fit(X[['contract_type']])
        self.encoder_nature_of_contract.fit(X[['nature_of_contract']])
        return self
  
  def transform(self, X):
        onehot_type = self.encoder_contract_type.transform(X[['contract_type']])
        onehot_nature = self.encoder_nature_of_contract.transform(X[['nature_of_contract']])
        onehot_type = pd.DataFrame(onehot_type, columns=['onehot_type'])
        onehot_nature = pd.DataFrame(onehot_nature, columns=['onehot_nature_1', 'onehot_nature_2'])
        X = X.join(onehot_nature).join(onehot_type)
        X.drop(['contract_type', 'nature_of_contract'], axis=1, inplace=True)
        return X

class TidyTitle(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.original_columns = None


    def fit(self, X ,y=None):
        self.original_columns = X.columns
        return self


    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, self.original_columns)
        X['title'] = X['title'].str.split(":").str.get(1)
        return X

def split_labels(data):
  """
  Parameters:
  data: the column of the dataframe with the target variable

  Returns:
  train_y: numpy array of labels. 1 if a contract satisfies a label, 0 otherwise. Each contract can have more than one label
  """
  rows = len(data)
  train_y = np.zeros([rows, 9])
  for row, label in enumerate(data):
      for col, l in enumerate(label):
          train_y[row][col] = 1. if l == '1' else 0.
  return train_y 

def split_data(data):
  """
  this function splits the data into two categories: word tokens and onehot data,
  then returns a list of two elements: one for each data type.
  
  Parameters:
  data: dataframe

  Returns:
  token_ids: numpy array of token id values.
  categorical_data: numpy array of onehot data
  """
    categorical_data = data[data.columns[len(data.columns)-3:]]
    token_ids = data[data.columns[:len(data.columns)-3]]
    
    categorical_data = categorical_data.to_numpy(dtype=np.float64)
    token_ids = token_ids.to_numpy()

    return [token_ids, categorical_data]
    
