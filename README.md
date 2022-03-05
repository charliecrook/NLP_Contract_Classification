# Natural Language Processing: Contract Classification

## File Structure

The data pipline classes and functions are in PreprocessingPipeline.py, and the CNN is in ClassificationModel.py. Three functions are created to aid the training and evaluation stage, these are included TrainingFunctions.py.

## Overview and Abstract

I report the results of a multilabel classification problem with the aim of accurately classifying all relevant categories for each tender contract in the dataset, which was provided by BIP Solutions.

As there was almost no correlation between the value or date and the labels, I identified the most important information was held within the text fields. I used a convolutional neural netwowrk with word token vectorisation as inspired by Yoon Kim's model (Convolutional Neural Networks for Sentence Classification. EMNLP. 2014). The model achieved a score of 0.919 on Kaggle.

## Data Preprocessing

The model takes the columns 'nature_of_contract', 'contract_type', 'title' and 'description'. The 'title' and 'description' columns are combined to a single string. Any NaN values are replaced with the most common values. The 'nature_of_contract' and 'contract_type' are converted to one-hot vectors. Finally word tokens are generated for the string values in the combined 'title' and 'description' column.

## Model: Convolutional Neural Network with Vectorisation

This model uses vectorisation embedding. An n-dimensional vector is created for each input token (corresponding to each word). A one-dimensional filter is convolved with the token vectors. One vector represents each token, and the convolutional filter passes over each vector in the set. The model also includes max pooling, dropout and fullly connected layers downstream. 

The model that I have used for this task takes in two inputs. The first input is an array of word tokens of the 'title' and 'description' columns. This is passed into an embedding layer, which converts the tokens to fixed-sized vectors of dimension 20. The vectors are passed into two convolutional layers with 30 filters each, and filter size of 2 and 3 respectively. The hyperparameters were chosen to provide the best performance while keeping computation times manageable. The output of these layers is passed into a max pooling and dropout layer. The model then receives a new input consiting of three one-hot vectors representing 'nature_of_contract' and 'contract_type'. 

## Approach to training and testing

The model was trained using the Adam optimiser with binary cross entropy loss. The main regularisation technique employed is early-stopping: each epoch the model calculates the loss on the validation set, if this does not decrease for 5 consecutive epochs, then training is interrupted to avoid overfitting of the training set. 
