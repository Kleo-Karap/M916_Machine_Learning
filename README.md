# M916_Machine_Learning
M.Sc. in Language Technology M916- Machine Learning assignments

## Linear Regression
In this task we explore Linear Regression in a common introductory dataset: The Boston Housing Dataset.
Dataset Overview
The dataset consists of 506 samples and 13 features/regressors. We want to use this data in order to
construct an estimator to be able to predict the price of a house given the features. You can load the
dataset and create a single panda dataframe for both the dataset data and the target variable.

Tasks:

**Train the model**
1. Use the train test split sklearn function to split the data into train and test sets using a ratio of
your choice, e.g. 90% training, 10% test, e.t.c.
2. Train a linear regression model on the training data using the normal equations. Do not use the
built in implementation of the sklearn package, but instead employ the coefficient estimates that
you derived in the previous section.

**Evaluate the results**
1. Evaluate the model, using the statistics introduced in Linear regression lecture for both training
and test data (RSE, R2).
2. Is there a difference between the produced RSE and R2 statistics of the train and test sets? If yes,why do you think this happens?
3. Choose the following training/test ratios, 80/20, 70/30, 50/50 for train test split and repeat thetraining process. Show the corresponding RSE and R2 metrics for each split and explain the potential differences among the obtained estimators.
## Logistic Regression/ Naive Bayes / Support Vector Machines
Heart Disease Dataset: Overview

The dataset consist of 303 samples and 13 features/regressors. We want to use this data in order to
construct a classifier to define whether a patient suffers from heart diseases and is likely to have a
heart attack. You can load the dataset and create a single panda dataframe for both the dataset data
and the target variable.

Tasks
1. **Prepare dataset**: Use the train test split sklearn function to split the data into train and
test sets using a ratio of your choice, e.g. 90% training, 10% test, e.t.c.
2. **Train a logistic regression model (LR)**: you are free to use whichever python package
you desire.
3. **Train a Support Vector Machine Classifier (SVM)**: try a linear and an radial basis
function kernel. For both cases try out C = [1, 10, 20], (gamma values can be chosen automatically).
4. **Train a Naive Bayes Classifier (NB)** : you are free to use whichever python package you
desire.
5. **Evaluate the results**: Use the metrics introduced in the lectures (accuracy, Confusion
Matrix). Is there a difference among the trained models LR, SVM and NB on the test set? If yes,why
do you think this happens?

## Neural Networks (ANNs and CNNs)
Mnist - Handwritten Dataset: Overview

The MNIST database of handwritten digits, is a large database of handwritten digits that is commonly
used for training various image processing systems (Figure 1). The database is also widely used for
training and testing in the field of machine learning. It was created by ”re-mixing” the samples from
NIST’s original datasets. It has a training set of 60,000 examples, and a test set of 10,000 examples.
It is a subset of a larger set available from NIST. The digits have been size-normalized and centered
in a fixed-size image.

Tasks
- **Load Dataset**: Use the Keras in-built module to load the dataset. Normalize both train and
test image values from [0,255] to [0,1]. From the dataset is pre-split in train and test set.
- **Artificial Neural Network (ANN)**: Train a neural network that consists of two
fully connected (Dense) layers. The first should have 256 neurons and the latter one 128 neurons.
Use ReLU activation function for every hidden layer. Finally use a fully connected layer (output
layer) with ten (10) neurons (one for each class) with softmax as its activation function. Use a
validation set from the training set (10% of the training dataset for validation)

- **Convolutional Neural Network (CNN)**: Train a CNN that consists of a convolutional layer with 64 kernels, a convolutional neural layer with 32 kernels and a fully connected layer with 128 neurons. All convolutional layer should have a 3. × 3 kernel size. The dense layer again should have ReLU as its activation function. Finally use a fully connected layer (output layer) with ten (10) neurons (one for each class) with softmax as its activation function. Use a
validation set from the train set (10% for validation).Try to use pooling (max pooling) after each convolutional layer. Inspect the results.

- **Loss/Accuracy**: For both the cases (ANN and CNN) plot the Accuracy and Loss for
vs the number of training epochs for the training and the validation data. Discuss on overfitting
and underfitting issues that may occur.
- **Evaluate the results**: Use accuracy to evaluate the performance of the trained NN
and CNN models and depict the corresponding confusion matrices. Is there a difference among
the trained models NN and CNN on the testset? If yes,why do you think this happens?
## References
- https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
- https://archive.ics.uci.edu/dataset/45/heart+disease

