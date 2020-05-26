""" utilityFunctions.py
    This script is used as a library, it contains several functions that are used for
    implemeting k-nearest neighbors to create a model for predicting.

    Author:         Jonathan Nunez Rdz.
    Institution:    Universidad de Monterrey
    First Created:  Wed 13 May, 2020
    Email:          jonathan.nunez@udem.edu // jonathannunezr1@gmail.com
"""

import pandas as pd
import numpy as np
import sys

def clean_column(data, mode, **kwargs):
    data = data.T
    column = np.zeros(shape= (data.shape[0], 1))
    labels = list()

    if mode == 'training':
        for i in range(data.shape[0]):
            if (data[i,0] not in labels):
                labels.append(data[i,0])
            column[i,0] = labels.index(data[i,0])
        
        return column[:,0], labels

    if mode == 'testing':
        labels = kwargs.get('labels')
        for i in range(data.shape[0]):
            column[i,0] = labels.index(data[i,0])
        
        return column[:,0]
    
    if mode == 'y_training':
        labels.append('Not Placed')
        labels.append('Placed')
        for i in range(data.shape[0]):
            column[i,0] = labels.index(data[i,0])

        return column[:,0], labels
    

"""load data from input csv by user"""
def load_data(file, display, will_scale_x, training_size):
    try:
        data = pd.read_csv(file)
    except:
        print("-"*40)
        print("'./{}' file directory doesn't exist".format(file))
        print("-"*40)
        exit(1)
    
    # split data into training and testing
    labels = np.array([data.columns.values])[0, 1:-1]
    training_data = data.sample(frac=training_size)
    testing_data = data.drop(training_data.index)

    if(display == 1):
        print('-'*40)
        print("training data set")
        print('-'*40)
        print(training_data)
        print('-'*40)
        print("testing data set")
        print('-'*40)
        print(testing_data)

    # split training data into x's and y
    x_training = pd.DataFrame.to_numpy(training_data.iloc[:, 1:-2])
    y_training = pd.Series.to_numpy(training_data.iloc[:, -2]).reshape(x_training.shape[0],1)

    # split testing data into x's and y
    x_testing = pd.DataFrame.to_numpy(testing_data.iloc[: , 1:-2])
    y_testing = pd.Series.to_numpy(testing_data.iloc[:, -2]).reshape(x_testing.shape[0], 1)

    # clean data columns
    labels_list = list()
    for i in range(x_training.shape[1]):
        if type(x_training[0,i]) == str:
            x_training[:,i], new_labels = clean_column(np.array([x_training[:,i]]), 'training')
            labels_list.append(new_labels)

    for i in range(y_training.shape[1]):
        if type(y_training[0,i]) == str:
            y_training[:,i], new_labels = clean_column(np.array([y_training[:,i]]), 'y_training')
            labels_list.append(new_labels)

    label_index = 0
    for i in range(x_testing.shape[1]):
        if type(x_testing[0,i]) == str:
            x_testing[:,i] = clean_column(np.array([x_testing[:,i]]), 'testing', labels=labels_list[label_index])
            label_index += 1

    for i in range(y_testing.shape[1]):
        if type(y_testing[0,i]) == str:
            y_testing[:,i] = clean_column(np.array([y_testing[:,i]]), 'testing', labels=labels_list[label_index])
            label_index += 1

    # print(pd.DataFrame(data=x_training, columns=labels[0:-1]))
    # print(pd.DataFrame(data=y_training, columns=[labels[-1]]))

    # feature scalling for x if specified
    if(will_scale_x == 1):
        # feature scalling for training
        x_training_scaled, mean, deviation = normalize_data(x_training, 'training')

        # feature scalling for testing
        x_testing_scaled = normalize_data(x_testing, 'testing', mean=mean, deviation=deviation)       

        # display option
        if(display == 1):
            print('-'*40)
            print("training features scaled")
            print('-'*40)
            for i in range(x_training_scaled.shape[0]):
                print(x_training_scaled[i])
            print('-'*40)
            print("testing features scaled")
            print('-'*40)
            for i in range(x_testing_scaled.shape[0]):
                print(x_testing_scaled[i])

        # return the training data and testing data scaled
        return x_training_scaled, y_training, x_testing_scaled, y_testing, labels, labels_list
    else:
        # return the training data and testing data
        return x_training, y_training, x_testing, y_testing, labels, labels_list

"""scale x_training to avoid conflicts"""
def normalize_data(x, mode, **kwargs):
    if(mode == 'training'):
        x_scaled = np.zeros_like(x)
        mean = np.zeros(shape = (x_scaled.shape[1], 1))
        deviation = np.zeros(shape = (x_scaled.shape[1], 1))

        for i in range(x.shape[1]):
            col = np.zeros_like(x)
            for j in range(x.shape[0]):
                col[j] = x[j][i]            
            mean[i] = col.mean()
            deviation[i] = col.std()
            for j in range(x.shape[0]):
                x_scaled[j][i] = (x[j][i] - col.mean()) / col.std()

        return x_scaled,mean,deviation
    
    if(mode == 'testing'):
        defined_mean = kwargs.get('mean')
        defined_deviation = kwargs.get('deviation')
        x_scaled = np.zeros_like(x)

        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                x_scaled[j][i] = (x[j][i] - defined_mean[i]) / defined_deviation[i]
        
        return x_scaled

# """return an array that contains (in the columns) the distances for each testing point"""
# def compute_euclidean_distances(x_training, x_testing):
#     euclidean_distances = np.zeros(shape = (x_training.shape[0],0))
    
#     for i in range(x_testing.shape[0]):
#         testing_point = x_testing[i,:]
#         euclidean_distance = compute_distance(x_training, testing_point)
#         euclidean_distances = np.concatenate((euclidean_distances, euclidean_distance), axis=1) 

#     return euclidean_distances

# """return the euclidean distance for a specific testing point"""
# def compute_distance(x_training, testing_point):
#     distance = np.zeros(shape = (x_training.shape[0], 1))
#     for i in range(x_training.shape[0]):
#         for j in range(x_training.shape[1]):
#             distance[i,0] += (x_training[i,j] - testing_point[j])**2
#         distance [i,0] = np.sqrt(distance[i,0])
#     return distance

# """return an array that contains the probabilities of belonging to either class 0 or 1"""
# def compute_conditional_probabilities(euclidean_distances, y_training, k):
#     nearest = np.zeros(shape = (euclidean_distances.shape[1], k))
#     for i in range(euclidean_distances.shape[1]):
#         euclidean_distance = np.array([euclidean_distances[:,i]])
#         euclidean_distance = np.concatenate((euclidean_distance.T, y_training), axis=1)
#         euclidean_distance = euclidean_distance[euclidean_distance[:,0].argsort()]
#         for j in range(k):
#             nearest[i,j] = euclidean_distance[j,1]

#     conditional_probalilities = np.zeros(shape = (nearest.shape[0], 2))

#     for i in range(nearest.shape[0]):
#         count_positives = 0
#         count_negatives = 0
#         for j in range(nearest.shape[1]):
#             if nearest[i,j] == 1: count_positives += 1
#             else: count_negatives += 1
#         conditional_probalilities[i, 0] = count_positives / k
#         conditional_probalilities[i, 1] = count_negatives / k

#     return conditional_probalilities

# """predict based on the conditional probabilities"""
# def predict(conditional_probalilities):
#     predictions = np.zeros(shape = (conditional_probalilities.shape[0], 1))
#     for i in range(conditional_probalilities.shape[0]):
#         if conditional_probalilities[i,0] >= conditional_probalilities[i,1]: predictions[i,0] = 1
#         else: predictions[i,0] = 0
    
#     return predictions

# """get the true positives, truw negatives, false positives, and false negatives as an array"""
# def get_confussion_matrix(predictions, y_testing):
#     confusion_matrix = np.zeros([2,2])

#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0

#     for i in range(predictions.shape[0]):
#         if predictions[i,0] == 1 and y_testing[i,0] == 1: TP += 1
#         elif predictions[i,0] == 0 and y_testing[i,0] == 0: TN += 1
#         elif predictions[i,0] == 1 and y_testing[i,0] == 0: FP += 1
#         else: FN += 1

#     confusion_matrix[0][0] = TP
#     confusion_matrix[1][1] = TN
#     confusion_matrix[0][1] = FP
#     confusion_matrix[1][0] = FN

#     return confusion_matrix

# """print to the console the performance metrics"""
# def print_performance_metrics(confusion_matrix):
#     TP = confusion_matrix[0][0]
#     TN = confusion_matrix[1][1]
#     FP = confusion_matrix[0][1]
#     FN = confusion_matrix[1][0]

#     accuracy = (TP + TN)/(TP + TN + FP + FN)
#     precision = TP /(TP + FP)
#     recall = TP/(TP + FN)
#     specificity = TN/(TN + FP)
#     f1_score = 2*(precision * recall)/(precision + recall)

#     print('-'*40)
#     print('confusion matrix')
#     print('-'*40)
#     print('true positives\t\t=> {}'.format(TP))
#     print('true negatives\t\t=> {}'.format(TN))
#     print('false positives\t\t=> {}'.format(FP))
#     print('false negatives\t\t=> {}'.format(FN))
#     print('-'*40)
#     print('scores')
#     print('-'*40)
#     print('accuracy\t\t=> {}'.format(accuracy))
#     print('precision\t\t=> {}'.format(precision))
#     print('recall\t\t\t=> {}'.format(recall))
#     print('specificity\t\t=> {}'.format(specificity))
#     print('f1 score\t\t=> {}'.format(f1_score))

# """print as a pandas dataframe"""
# def print_features(labels, x_testing, conditional_probalilities):
#     labels = np.array([labels[0:-1]])
#     labels = np.concatenate((labels, [['Prob. Diabetes', 'Prob. No Diabetes']]), axis=1)

#     x_testing = np.concatenate((x_testing, conditional_probalilities), axis=1)

#     df = pd.DataFrame(data=x_testing, columns=labels[0,:])
    
#     print(df)