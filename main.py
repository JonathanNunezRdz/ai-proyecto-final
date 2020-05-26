""" main.py
    This script implements the euclideam distance to get the model for 
    k-nearest neighbors and predict based on the probabilities of belonging
    to a certain class.

    Author:         Jonathan Nunez Rdz.
    Institution:    Universidad de Monterrey
    First Created:  Wed 13 May, 2020
    Email:          jonathan.nunez@udem.edu // jonathannunezr1@gmail.com
"""

def main():
    #   import standard libraries
    import numpy as np
    import pandas as pd
    import sys
    import time

    #   import functions
    import utilityFunctions as uf

    print("--- the program has started, please wait ---")
    start_time = time.time()

    #   define the name of the flie to read the data from
    csv = 'campus_placement.csv'
    #   define the training data size, testing data size will be calculated accordingly
    training_size = 0.80

    #   display is required and is used for whether or not to print the training data set:
    #
    #   display = 1, will print to the console
    #   display = 0, will not print
    display = 0
    #   -------------------------------------------------------------------------------------------
    #   will_scale is required and it will define whether or not to implement feature 
    #   scalling for the testing data 
    # 
    #   will_scale = 1, will implement feature scalling 
    #   wii_scale = 0, will not
    will_scale_x = 1
    #   -------------------------------------------------------------------------------------------
    #   uf.load_data will read the determined csv (must be local) with pandas and divide tha data to x_training 
    #   and y_training, the data will be divided in according to the specified training_size in line 28.
    x_training, y_training, x_testing, y_testing, labels, labels_list = uf.load_data(csv, display, will_scale_x, training_size)

    print(labels_list)
    sys.exit()


    #   calculate the distances for each of the x_testing row with all of the rows of X_training
    euclidean_distances = uf.compute_euclidean_distances(x_training, x_testing)

    #   k defines how many rows it will compare to get the probabilities for the classes
    k = 20
    #   -------------------------------------------------------------------------------------------
    #   get the probabilities, with the distances of the x_testing, of belonging to a class, in this case, 0 or 1
    conditional_probabilities = uf.compute_conditional_probabilities(euclidean_distances, y_training, k)


    #   make the predictions based on the conditional probabilities
    predictions = uf.predict(conditional_probabilities)


    #   return a matrix with the true positives, true negatives, false positives, and false negatives for the predictions made
    confusion_matrix = uf.get_confussion_matrix(predictions,y_testing)

    
    #   print_performance_metrics will print the accuracy, precision, recall, specificity, and f1 score based on the confusion matrix
    uf.print_performance_metrics(confusion_matrix)


    uf.print_features(labels, x_testing, conditional_probabilities)

if __name__ == "__main__":
    main()