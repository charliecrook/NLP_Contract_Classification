
import keras

class CNNVectorisationModel(keras.Model):
  def __init__(self, vocabulary_size=27000,
               embedding_dimensions=20, 
               filters=30, 
               activation="relu", **kwargs):
    
    super().__init__(**kwargs)

    self.embedded = keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dimensions, input_length=512)
    self.conv1 = keras.layers.Conv1D(filters=filters,kernel_size=2, padding="valid", activation=activation)
    self.conv2 = keras.layers.Conv1D(filters=filters,kernel_size=3, padding="valid", activation=activation)
    self.pooling = keras.layers.GlobalMaxPool1D()
    self.dropout = keras.layers.Dropout(rate=0.2)
    self.concat = keras.layers.Concatenate(axis=1)
    self.dense1 = keras.layers.Dense(512, activation=activation)
    self.dropout2 = keras.layers.Dropout(rate=0.2)
    self.dense2 = keras.layers.Dense(256, activation=activation)
    self.dropout3 = keras.layers.Dropout(rate=0.2)
    self.output_ = keras.layers.Dense(9, activation='sigmoid')
 
  def call(self, inputs):
    
    tokens, categories = inputs   
    embedded = self.embedded(tokens)
    conv1 = self.conv1(embedded)
    conv2 = self.conv2(conv1)
    pooling = self.pooling(conv2)
    dropout = self.dropout(pooling)
    concat = self.concat([dropout, categories])
    dense1 = self.dense1(concat)
    dropout2 = self.dropout2(dense1)
    dense2 = self.dense2(dropout2)
    dropout3 = self.dropout3(dense2)
    output_ = self.output_(dropout3)

    return output_