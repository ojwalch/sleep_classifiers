import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix,recall_score,cohen_kappa_score,accuracy_score
import os.path

# Set flags for what mode to run
RUN_SW = 0
RUN_REM = 1
RUN_ALL = 2
verbose = 0

# Subject identifiers
FULL_SET = [1,2,4,5,6,7,8,9,10,11,12,14,15,16,19,20,22,23,25,27,28,29,30,32,33,34,35,38,39,41,42]
print(FULL_SET)

# Make dictionary with training and testing subdivisions
def make_train_test_dict(trials,test_frac=0.3):
    tt_dict = {}
    for run in range(0,trials):
        tt_dict[run] = train_test_sets(test_frac)
    return tt_dict

# Creates training and testing sets
def train_test_sets(test_frac):
    train_set = FULL_SET
    random.shuffle(train_set)
    test_ind = int(np.round(test_frac*len(train_set)))
    test_set = train_set[0:test_ind]
    train_set = train_set[test_ind + 1:]
    
    return train_set, test_set

# Prints performance metrics to LaTeX array
def data_to_line(name,data,latex=False):
    if latex:
        return name + ' & ' + str(round(np.mean(data),2)) + ' (' + str(round(np.std(data),2)) + ') & ' + str(round(np.min(data),2)) + '-' + str(round(np.max(data),2)) + '\\\\'
    else:
        return name + ',' + str(round(np.mean(data),2)) + ' (' + str(round(np.std(data),2)) + '),' + str(round(np.min(data),2)) + '-' + str(round(np.max(data),2))


# Save array to an XLSX spreadsheet - for exporting tables. TODO: Use or delete
def save_array(fname,array):
    df = pd.DataFrame (array)
    df.to_excel(fname, index=False)

# Converts val_y to [0,1] (wake/sleep), [0,1,2] (wake, NREM, REM), or -4 to 1 for all stages
def process_raw_scores(val_y, run_flag):
    val_y_processed = np.array([])
    
    # If you want all sleep stages classified:
    if run_flag == RUN_ALL:
        for epoch in val_y:
            if epoch == 5: # Set REM (5) to 1
                val_y_processed = np.append(val_y_processed,1)
            else: # Otherwise, negate
                val_y_processed = np.append(val_y_processed,-1*epoch)

        return val_y_processed

    # Otherwise, go epoch by epoch
    for epoch in val_y:
        if (run_flag == RUN_REM):
            if epoch == 0:
                val_y_processed = np.append(val_y_processed,0) # Wake
            else:
                if epoch == 5:
                    val_y_processed = np.append(val_y_processed,2) # REM
                else:
                    val_y_processed = np.append(val_y_processed,1) # NREM
        else: # Just sleep/wake
            if epoch == 0:
                val_y_processed = np.append(val_y_processed,0) # Wake
            else:
                val_y_processed = np.append(val_y_processed,1) # Sleep

    return val_y_processed


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
    features = ['Motion','HR','Clock','Time','CircModel']
    for f in features:
        if(feature_set[f]):
            string = string + f
    return string


# Scales selected data columns TODO: Use or remove
def scale_data_dictionary(data_dict,scale_width):
    all_features = np.array([])
    
    # For all subjects in the data dictionary:
    for key in data_dict:
        full_features = data_dict[key][1]
        if(len(np.shape(full_features)) < 2):
            full_features = np.transpose([full_features])
    
        if(np.shape(all_features)[0] == 0):
            all_features = full_features
        else:
            all_features = np.vstack((all_features,full_features))

    # Compute mean and standard deviation across all subjects
    cols = np.shape(all_features)[1]
    col_means = np.mean(all_features, axis=0)
    col_std = np.std(all_features, axis=0)

    # Normalize each subject individually
    for key in data_dict:
        full_features = data_dict[key][1]
        if(len(np.shape(full_features)) < 2):
            full_features = np.transpose([full_features])

        for col in range(0,scale_width):
            full_features[:,col] = (full_features[:,col] - col_means[col])/(col_std[col])

        data_dict[key][1] = full_features
    return data_dict


# Extract features from the data dictionary
def get_features(subject_id,feature_set,data_dict):
    subject_data = data_dict[str(subject_id)]
    return subject_data[0], subject_data[1]


# Builds feature vector from feature_set
def build_data_dictionary(feature_set):
    if verbose:
        print('Building data dictionary...')

    
    data_dict = {} # Holder for subject data

    # Loop over subjects, loading data
    for subject_id in FULL_SET:
        path = '../data/features/' + str(subject_id)
        full_features = np.array([])
    
        if feature_set['Motion']:
            motion_features = np.genfromtxt(path + '_motion_feat.csv', delimiter=',')
            if(len(np.shape(motion_features)) < 2):
                motion_features = np.transpose([motion_features])
            if(np.shape(full_features)[0] == 0):
                full_features = motion_features
            else:
                full_features = np.hstack((full_features,motion_features))

        if feature_set['HR']:
            hr_features = np.genfromtxt(path + '_hr_feat.csv', delimiter=',')
            if(len(np.shape(hr_features)) < 2):
                hr_features = np.transpose([hr_features])
            if(np.shape(full_features)[0] == 0):
                full_features = hr_features
            else:
                full_features = np.hstack((full_features,hr_features))

        if feature_set['Time']:
            t_features = np.genfromtxt(path + '_time_feat.csv', delimiter=',')
            if(len(np.shape(t_features)) < 2):
                t_features = np.transpose([t_features])
            if(np.shape(full_features)[0] == 0):
                full_features = t_features
            else:
                full_features = np.hstack((full_features,t_features))

        if feature_set['Clock']:
            circ_features = np.genfromtxt(path + '_clock_feat.csv', delimiter=',')
            if(len(np.shape(circ_features)) < 2):
                circ_features = np.transpose([circ_features])
            if(np.shape(full_features)[0] == 0):
                full_features = circ_features
            else:
                full_features = np.hstack((full_features,circ_features))

        if feature_set['CircModel']:
            if(os.path.isfile(path + '_circ_model_feat.csv')):
                cm_features = np.genfromtxt(path + '_circ_model_feat.csv', delimiter=',')
            else:
                cm_features = np.genfromtxt(path + '_clock_feat.csv', delimiter=',')

            if(len(np.shape(cm_features)) < 2):
                cm_features = np.transpose([cm_features])


            if np.isnan(np.sum(cm_features)): # Check to make sure nothing went wrong with ODE
                print('NaN data detected in subject ' + subject_id)

            if(np.shape(full_features)[0] == 0):
                full_features = cm_features
            else:
                full_features = np.hstack((full_features,cm_features))

        score_features = np.transpose([np.genfromtxt(path + '_score_feat.csv', delimiter=',')])
        
        # Add and scores features to data dictionary
        data_dict[str(subject_id)] = [score_features,full_features]
    
    if verbose:
        print('Data dictionary complete...')
    
    return data_dict


# Interpolate to find classification at a given true positive threshold
def thresh_interpolation(fpr, tpr, thresholds, val_predictions_proba, val_y):
    
    num_samples = np.shape(val_predictions_proba)[0]
    val_predictions = np.zeros((num_samples,1))

    all_scores = [] # Holder for performance scores at difference true positive thresholds
    for tp_thresh in [0.8, 0.9, 0.93, 0.95]:
        thresh = np.interp(tp_thresh,tpr,thresholds)
        
        sleep_inds = np.where(val_predictions_proba[:,1] > thresh)[0]
        wake_inds = np.where(val_predictions_proba[:,1] <= thresh)[0]
        
        val_predictions[sleep_inds,0] = 1
        val_predictions[wake_inds,0] = 0
                
        all_scores.append(np.array([accuracy_score(val_y,val_predictions), recall_score(val_y,val_predictions,pos_label=0), recall_score(val_y,val_predictions), cohen_kappa_score(val_y,val_predictions),auc(fpr, tpr)]))
    
    return all_scores
