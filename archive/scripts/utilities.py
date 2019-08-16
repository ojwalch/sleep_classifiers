import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, cohen_kappa_score, accuracy_score
import os.path

# Set flags for what mode to run
RUN_SW = 0
RUN_REM = 1
RUN_ALL = 2
verbose = 0

# Subject identifiers
FULL_SET = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 22, 23, 25, 27, 28, 29, 30, 32, 33, 34, 35, 38, 39,
            41, 42]
print(FULL_SET)


def make_train_test_dict(trials, test_frac=0.3):
    """
        Make dictionary with training and testing subdivisions

        Args:
            trials (int): Number of training/testing splits to generate
            test_frac (float) : Fraction of data to use for testing

        Returns:
           dict : Maps run count to training and testing set
        """

    train_test_dict = {}
    for run in range(0, trials):
        train_test_dict[run] = train_test_sets(test_frac)
    return train_test_dict


def train_test_sets(test_frac):
    """
        Creates training and testing sets

        Args:
            test_frac (float) : Fraction of data to use for testing

        Returns:
           [int],[int] : Array of subject numbers for training and for testing
        """

    train_set = FULL_SET
    random.shuffle(train_set)
    test_ind = int(np.round(test_frac*len(train_set)))
    test_set = train_set[0:test_ind]
    train_set = train_set[test_ind + 1:]
    
    return train_set, test_set


# Prints performance metrics to LaTeX array
def data_to_line(name, data, latex=False):
    if latex:
        return name + ' & ' + str(round(np.mean(data),2)) + ' (' + str(round(np.std(data),2)) + ') & ' \
               + str(round(np.min(data), 2)) + '-' + str(round(np.max(data), 2)) + '\\\\'
    else:
        return name + ',' + str(round(np.mean(data),2)) + ' (' + str(round(np.std(data),2)) + '),' \
               + str(round(np.min(data), 2)) + '-' + str(round(np.max(data), 2))


# Save array to an XLSX spreadsheet - for exporting tables
# TODO Use or delete
def save_array(fname, array):
    df = pd.DataFrame(array)
    df.to_excel(fname, index=False)


def process_raw_scores(true_labels, run_flag):
    """
        Converts raw labels to [0,1] (wake/sleep), [0,1,2] (wake, NREM, REM), or -4 to 1 for all stages

        Args:
            true_labels ([int]) : Raw labels
            run_flag (int) : Type of classification to perform

        Returns:
           np.array : Processed labels
        """
    processed_labels = np.array([])
    
    if run_flag == RUN_ALL:         # Wake, N1-3, REM
        for epoch in true_labels:
            if epoch == 5: # Set REM (5) to 1
                processed_labels = np.append(processed_labels, 1)
            else: # Otherwise, negate
                processed_labels = np.append(processed_labels, -1*epoch)

        return processed_labels

    for epoch in true_labels:
        if run_flag == RUN_REM:   # Wake/NREM/REM
            if epoch == 0:
                processed_labels = np.append(processed_labels, 0)  # Wake
            else:
                if epoch == 5:
                    processed_labels = np.append(processed_labels, 2)  # REM
                else:
                    processed_labels = np.append(processed_labels, 1)  # NREM
        else:                    # Sleep/wake
            if epoch == 0:
                processed_labels = np.append(processed_labels, 0)  # Wake
            else:
                processed_labels = np.append(processed_labels, 1)  # Sleep

    return processed_labels


# Makes plot look nicer
def tidy_plot():
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


# Creates string from feature set
def string_from_features(feature_set):
    string = ''
    features = ['Motion', 'HR', 'Clock', 'Time', 'CircModel']
    for f in features:
        if feature_set[f]:
            string = string + f
    return string


# Extracts features and labels from data dictionary
def get_features(subject_id, data_dict):
    subject_data = data_dict[str(subject_id)]
    return subject_data[0], subject_data[1]


def build_data_dictionary(feature_set):
    """
        Builds data dictionary for all subjects given a feature set to test

        Args:
            feature_set (dict) : Feature set to build data dictionary from

        Returns:
           dict : Dictionary of subject data; keys are subject IDs
        """
    if verbose:
        print('Building data dictionary...')

    data_dict = {}  # Holder for subject data

    for subject_id in FULL_SET:             # Load data for all subjects
        path = '../data/features/' + str(subject_id)
        full_features = np.array([])
    
        if feature_set['Motion']:
            motion_features = np.genfromtxt(path + '_motion_feat.csv', delimiter=',')
            if len(np.shape(motion_features)) < 2:
                motion_features = np.transpose([motion_features])
            if np.shape(full_features)[0] == 0:
                full_features = motion_features
            else:
                full_features = np.hstack((full_features, motion_features))

        if feature_set['HR']:
            hr_features = np.genfromtxt(path + '_hr_feat.csv', delimiter=',')
            if len(np.shape(hr_features)) < 2:
                hr_features = np.transpose([hr_features])
            if np.shape(full_features)[0] == 0:
                full_features = hr_features
            else:
                full_features = np.hstack((full_features, hr_features))

        if feature_set['Time']:
            t_features = np.genfromtxt(path + '_time_feat.csv', delimiter=',')
            if len(np.shape(t_features)) < 2:
                t_features = np.transpose([t_features])
            if np.shape(full_features)[0] == 0:
                full_features = t_features
            else:
                full_features = np.hstack((full_features, t_features))

        if feature_set['Clock']:
            circ_features = np.genfromtxt(path + '_clock_feat.csv', delimiter=',')
            if len(np.shape(circ_features)) < 2:
                circ_features = np.transpose([circ_features])
            if np.shape(full_features)[0] == 0:
                full_features = circ_features
            else:
                full_features = np.hstack((full_features, circ_features))

        if feature_set['CircModel']:
            if os.path.isfile(path + '_circ_model_feat.csv'):
                circadian_model_features = np.genfromtxt(path + '_circ_model_feat.csv', delimiter=',')
            else:
                circadian_model_features = np.genfromtxt(path + '_clock_feat.csv', delimiter=',')

            if len(np.shape(circadian_model_features)) < 2:
                circadian_model_features = np.transpose([circadian_model_features])

            if np.isnan(np.sum(circadian_model_features)):  # Check to make sure nothing went wrong with ODE
                print('NaN data detected in subject ' + subject_id)

            if np.shape(full_features)[0] == 0:
                full_features = circadian_model_features
            else:
                full_features = np.hstack((full_features, circadian_model_features))

        score_features = np.transpose([np.genfromtxt(path + '_score_feat.csv', delimiter=',')])
        
        # Add and scores features to data dictionary
        data_dict[str(subject_id)] = [score_features,full_features]
    
    if verbose:
        print('Data dictionary complete...')
    
    return data_dict


# Interpolate to find classification at a given true positive threshold for tables
def thresh_interpolation(false_positive_rate_spread, true_positive_rate_spread, thresholds, class_probabilities,
                         true_labels):
    
    num_samples = np.shape(class_probabilities)[0]
    predicted_labels = np.zeros((num_samples, 1))

    all_scores = []  # Holder for performance scores at difference true positive thresholds
    for true_positive_interpolation_point in [0.8, 0.9, 0.93, 0.95]:
        thresh = np.interp(true_positive_interpolation_point, true_positive_rate_spread, thresholds)
        
        true_sleep_indices = np.where(class_probabilities[:, 1] > thresh)[0]
        true_wake_indices = np.where(class_probabilities[:, 1] <= thresh)[0]
        
        predicted_labels[true_sleep_indices, 0] = 1
        predicted_labels[true_wake_indices, 0] = 0
                
        all_scores.append(np.array([accuracy_score(true_labels, predicted_labels),
                                    recall_score(true_labels, predicted_labels, pos_label=0),
                                    recall_score(true_labels, predicted_labels),
                                    cohen_kappa_score(true_labels, predicted_labels),
                                    auc(false_positive_rate_spread, true_positive_rate_spread)]))
    
    return all_scores
