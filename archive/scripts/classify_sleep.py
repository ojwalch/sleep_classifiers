import numpy as np
import warnings
import time

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager
import matplotlib.cbook

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import class_weight

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

import utilities
import multiprocessing
import get_parameters

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

run_flag = utilities.RUN_SW
font_name = 'Arial'
verbose = False
NUM_REPS_TRAIN_TEST = 10
LOAD_PARAMS = False  # Load params saved from file
PRINT_TABLE = True  # Print LaTeX table for paper

# REM Binary search parameters
FALSE_POSITIVE_BUFFER = 0.001  # How close we have to be to the desired goal FP before it can be added to the average
MAX_ATTEMPTS_WAKE_BINARY_SEARCH = 50  # Number of times to try before quitting the binary search
NUM_FALSE_POSITIVE_POINTS_REM = 20
REM_NREM_ACCURACY_DIFFERENCE = 1e-2  # How close we want NREM and REM accuracies to be
MAX_ATTEMPTS_NREM_REM_BINARY_SEARCH = 15

# Constants for plotting and tables
NUM_FALSE_POSITIVE_POINTS_PLOT = 100
FALSE_POSITIVE_INTERPOLATION_POINT_REM_NREM_TABLES = 0.6

METHOD_DICT = {'Random Forest': RandomForestClassifier(n_estimators=500, max_features=1.0, max_depth=10,
                                                       min_samples_split=10, min_samples_leaf=1),
               'Logistic Regression': LogisticRegression(penalty='l1', solver='liblinear', verbose=0),
               'KNeighbors': KNeighborsClassifier(),
               'MLP': MLPClassifier(activation='relu', hidden_layer_sizes=(30, 30, 30), max_iter=1000, alpha=0.01)}

feature_sets = [{'Motion': True, 'HR': False, 'Clock': False, 'Time': False, 'CircModel': False},
                {'Motion': False, 'HR': True, 'Clock': False, 'Time': False, 'CircModel': False},
                {'Motion': True, 'HR': True, 'Clock': False, 'Time': False, 'CircModel': False},
                {'Motion': True, 'HR': True, 'Clock': False, 'Time': False, 'CircModel': True}]

cases = ['Motion only', 'HR only', 'Motion and HR', 'Motion, HR, Clock']

colors = [sns.xkcd_rgb["denim blue"],
          sns.xkcd_rgb["yellow orange"],
          sns.xkcd_rgb["medium green"],
          sns.xkcd_rgb["pale red"]]

global train_test_dict
global description


def train_and_test_model(training_subjects, testing_subjects, method_key, classifier, feature_set, data_dict,
                         save_to_file=False):
    """
        Trains and tests model for given feature set and classifier.
        
        Args:
            training_subjects ([int]): Subject IDs in training set
            testing_subjects ([int]): Subject IDs in testing set
            method_key (str): Key for classifier
            classifier : Classifier object
            feature_set (dict): Feature set to test
            data_dict (dict): Dictionary to look up subject training and testing data
            save_to_file (bool) : Flag if want to save probabilities to file

        Returns:
            [int]: ground truth labels
            np.array : predicted labels
            np.array : class prediction probabilities
        """

    classifier_abbrev = str(classifier)[0:4]
    save_name = 'parameters/' + classifier_abbrev + utilities.string_from_features(feature_set) + '_params.npy'

    if LOAD_PARAMS or method_key == 'MLP':  # TODO: Faster parameter searching with MLP
        params = np.load(save_name).item()
    else:
        params = get_parameters.find_best(method_key, feature_set, training_subjects)
        np.save(save_name, params)

    classifier.set_params(**params)

    training_set_features = np.array([])
    training_set_true_labels = np.array([])
    testing_set_features = np.array([])
    testing_set_true_labels = np.array([])

    # Get labels and features for training and testing sets
    for subject in training_subjects:
        scores_by_epoch, features_by_epoch = utilities.get_features(subject, data_dict)

        if np.shape(training_set_features)[0] == 0:
            training_set_features = features_by_epoch
            training_set_true_labels = scores_by_epoch
        else:
            training_set_features = np.vstack((training_set_features, features_by_epoch))
            training_set_true_labels = np.vstack((training_set_true_labels, scores_by_epoch))

    for subject in testing_subjects:
        scores_by_epoch, features_by_epoch = utilities.get_features(subject, data_dict)
        if np.shape(testing_set_features)[0] == 0:
            testing_set_features = features_by_epoch
            testing_set_true_labels = scores_by_epoch
        else:
            testing_set_features = np.vstack((testing_set_features, features_by_epoch))
            testing_set_true_labels = np.vstack((testing_set_true_labels, scores_by_epoch))

    # Convert raw labels to 0/1 or 0-2
    training_set_true_labels = utilities.process_raw_scores(training_set_true_labels, run_flag)
    testing_set_true_labels = utilities.process_raw_scores(testing_set_true_labels, run_flag)

    # Set class weights for those methods that allow them
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(training_set_true_labels),
                                                      y=training_set_true_labels)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    if len(class_weights) > 2:  # Handles wake/NREM/REM case
        class_weight_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

    classifier.class_weight = class_weight_dict

    # # Debug-only: Uncomment to reverse the training/testing order, and test Apple Watch data on MESA-trained models
    # classifier = np.load('trained_models/' + classifier_abbrev +
    # utilities.string_from_features(feature_set) + '_trained_modelMESA.npy').item()

    # Fit model to training data, get class predictions and class probabilities
    classifier.fit(training_set_features, training_set_true_labels)
    predicted_labels = classifier.predict(testing_set_features)
    class_probabilities = classifier.predict_proba(testing_set_features)

    # Save trained model to use for testing MESA cohort
    save_name = 'trained_models/' + classifier_abbrev + \
                utilities.string_from_features(feature_set) + '_trained_model.npy'
    np.save(save_name, classifier)

    # Optional; save to file for Kalman filter and print performance metrics
    if save_to_file:
        np.savetxt('sleep_modeling/' + str(testing_subjects[0]) + '.csv',
                   classifier.predict_proba(testing_set_features), delimiter=',')
        np.savetxt('sleep_modeling/' + str(testing_subjects[0]) + '_classes.csv',
                   testing_set_true_labels, delimiter=',')
        np.savetxt('sleep_modeling/' + str(testing_subjects[0]) + '_predicted_classes.csv',
                   predicted_labels, delimiter=',')

        true_positive_rate_for_interpolation = 0.85
        false_positive_rates, true_positive_rates, thresholds = roc_curve(testing_set_true_labels,
                                                                          class_probabilities[:, 1],
                                                                          pos_label=1, drop_intermediate=False)

        print('Subject ID: ' + str(testing_subjects[0]))
        print('False positive rate: ' + str(
            np.interp(true_positive_rate_for_interpolation, true_positive_rates, false_positive_rates)))
        print('True positive rate: ' + str(true_positive_rate_for_interpolation))
        print('\n\n')

    return testing_set_true_labels, predicted_labels, class_probabilities


def parallel_roc(trial_dictionary):
    """
        Calls training and testing model for ROC; allows parallelization

        Args:
            trial_dictionary (dict): All information needed to train and test the model for a classifier/feature set

        Returns:
            Performance metrics for the training/testing iteration

        """

    method = trial_dictionary['method']
    feature_set = trial_dictionary['feature_set']
    data_dict = trial_dictionary['data_dict']
    train_set = trial_dictionary['train_set']
    test_set = trial_dictionary['test_set']
    method_key = trial_dictionary['method_key']

    # Get ground truth, predictions, and class probabilities
    testing_set_true_labels, predicted_labels, class_probabilities = train_and_test_model(train_set, test_set,
                                                                                          method_key, method,
                                                                                          feature_set, data_dict)

    if run_flag == utilities.RUN_SW:  # If sleep/wake classification

        false_positive_rates, true_positive_rates, thresholds = roc_curve(testing_set_true_labels,
                                                                          class_probabilities[:, 1], pos_label=1,
                                                                          drop_intermediate=False)
        performance = utilities.thresh_interpolation(false_positive_rates, true_positive_rates, thresholds,
                                                     class_probabilities, testing_set_true_labels)
        return [false_positive_rates, true_positive_rates, thresholds, performance]

    else:  # If wake/NREM/REM classification

        false_positive_rates, true_positive_rate_average, nrem_accuracies, rem_accuracies, best_accuracies, \
        kappas_at_best_accuracies = roc_curve_rem(testing_set_true_labels, class_probabilities)
        return [false_positive_rates, true_positive_rate_average, nrem_accuracies, rem_accuracies, best_accuracies,
                kappas_at_best_accuracies]


def roc_curve_rem(true_labels, class_probabilities):
    """
        Make an "ROC curve for NREM/REM/wake classification" by looping over desired false positive rates
        and performing two binary searches: one for a wake threshold, and one to balance the accuracies of the REM
        and NREM classes

        Args:
            true_labels (np.array): Ground truth labels for tested epochs
            class_probabilities (np.array): Class probabilities for tested epochs

        Returns:
            false positive rates, average NREM/REM accuracies, individual REM/NREM accuracies, best accuracies
            found during the search, and kappas at best accuracies
        """

    goal_false_positive_spread = []  # Spread of targeted goal false positive rates
    for i in range(0, NUM_FALSE_POSITIVE_POINTS_REM):
        goal_false_positive_spread.append(i / (NUM_FALSE_POSITIVE_POINTS_REM * 1.0))

    goal_false_positive_spread = np.array(goal_false_positive_spread)

    # Holders for performance metrics
    false_positive_rate_spread = []
    true_positive_rate_spread = []
    accuracies = []
    kappas = []
    nrem_class_accuracies = []
    rem_class_accuracies = []

    start = time.time()

    true_wake_indices = np.where(true_labels == 0)[0]  # Indices where ground truth is wake
    true_nrem_indices = np.where(true_labels == 1)[0]  # Indices of ground truth NREM
    true_rem_indices = np.where(true_labels == 2)[0]  # Indices of ground truth REM

    # Get coverage over entire x-axis of ROC curve by repeating binary searches over a spread
    for goal_false_positive_rate in goal_false_positive_spread:

        false_positive_rate = -1
        binary_search_counter = 0

        # Search while we haven't found the target false positive rate
        while (false_positive_rate < goal_false_positive_rate - FALSE_POSITIVE_BUFFER
               or false_positive_rate >= goal_false_positive_rate + FALSE_POSITIVE_BUFFER) and binary_search_counter < MAX_ATTEMPTS_WAKE_BINARY_SEARCH:

            if binary_search_counter == 0:  # Start binary search conditions
                threshold_for_sleep = 0.5
                threshold_delta = 0.25
            else:  # Update threshold based on difference between goal and actual false positive rate
                if false_positive_rate < goal_false_positive_rate - FALSE_POSITIVE_BUFFER:
                    threshold_for_sleep = threshold_for_sleep - threshold_delta
                    threshold_delta = threshold_delta / 2
                if false_positive_rate >= goal_false_positive_rate + FALSE_POSITIVE_BUFFER:
                    threshold_for_sleep = threshold_for_sleep + threshold_delta
                    threshold_delta = threshold_delta / 2

            if goal_false_positive_rate == 1:  # Edge cases
                threshold_for_sleep = 0.0
            if goal_false_positive_rate == 0:
                threshold_for_sleep = 1.0

            predicted_sleep_indices = np.where(1 - np.array(class_probabilities[:, 0]) >= threshold_for_sleep)[0]

            predicted_labels = np.zeros(np.shape(true_labels))
            predicted_labels[predicted_sleep_indices] = 1  # Set locations of predicted sleep to 1

            predicted_labels_at_true_wake_indices = predicted_labels[true_wake_indices]

            # FPR: 1 - Wake scored as wake, a.k.a  1 - (Total true wake - true wake scored as sleep)/(Total true wake)
            number_wake_correct = len(true_wake_indices) - np.count_nonzero(predicted_labels_at_true_wake_indices)
            fraction_wake_correct = number_wake_correct / (len(true_wake_indices) * 1.0)
            false_positive_rate = 1.0 - fraction_wake_correct

            binary_search_counter = binary_search_counter + 1

            # # Uncomment for debugging:
            # print('Goal FP = ' + str(goal_false_positive_rate) + ' Thresh: ' + str(threshold_for_sleep) + ',
            # Delta: ' + str(threshold_delta) + ', False positive rate: ' + str(false_positive_rate) + ',
            # Count: ' + str(binary_search_counter))

        if binary_search_counter < MAX_ATTEMPTS_WAKE_BINARY_SEARCH:  # Checks we found our target false positive rate

            # Initial values for binary search
            smallest_accuracy_difference = 2  # Difference between NREM and REM accuracies
            true_positive_rate = 0
            rem_accuracy = 0
            nrem_accuracy = 0
            best_accuracy = -1
            kappa_at_best_accuracy = -1

            # Initial values for second threshold binary search
            count_thresh = 0
            threshold_for_rem = 0.5
            threshold_delta_rem = 0.5

            while count_thresh < MAX_ATTEMPTS_NREM_REM_BINARY_SEARCH and \
                    smallest_accuracy_difference > REM_NREM_ACCURACY_DIFFERENCE:

                count_thresh = count_thresh + 1

                for predicted_sleep_index in range(len(predicted_sleep_indices)):
                    predicted_sleep_epoch = predicted_sleep_indices[predicted_sleep_index]

                    if class_probabilities[predicted_sleep_epoch, 2] > threshold_for_rem:
                        predicted_labels[predicted_sleep_epoch] = 2  # Set to REM sleep
                    else:
                        predicted_labels[predicted_sleep_epoch] = 1  # Set to NREM sleep

                # Compute accuracy and kappa at this threshold during the search
                accuracy = accuracy_score(predicted_labels, true_labels)
                kappa = cohen_kappa_score(predicted_labels, true_labels)

                if accuracy > best_accuracy:  # Save if we've exceeded best accuracy
                    best_accuracy = accuracy
                    kappa_at_best_accuracy = kappa

                predicted_nrem_indices = np.where(predicted_labels == 1)[0]
                predicted_rem_indices = np.where(predicted_labels == 2)[0]

                # Compute NREM/REM accuracies -- number of true class epochs scored as class, divided by number in class
                correct_nrem_indices = np.intersect1d(predicted_nrem_indices, true_nrem_indices)
                correct_rem_indices = np.intersect1d(predicted_rem_indices, true_rem_indices)
                nrem_accuracy = len(correct_nrem_indices) / (1.0 * len(true_nrem_indices))
                rem_accuracy = len(correct_rem_indices) / (1.0 * len(true_rem_indices))
                true_positive_rate = (rem_accuracy + nrem_accuracy) / 2.0

                smallest_accuracy_difference = np.abs(nrem_accuracy - rem_accuracy)

                if rem_accuracy < nrem_accuracy:
                    threshold_for_rem = threshold_for_rem - threshold_delta_rem / 2.0
                else:
                    threshold_for_rem = threshold_for_rem + threshold_delta_rem / 2.0

                threshold_delta_rem = threshold_delta_rem / 2.0

            # Add found values to holders
            false_positive_rate_spread.append(false_positive_rate)
            true_positive_rate_spread.append(true_positive_rate)
            nrem_class_accuracies.append(nrem_accuracy)
            rem_class_accuracies.append(rem_accuracy)
            accuracies.append(best_accuracy)
            kappas.append(kappa_at_best_accuracy)

    end = time.time()

    if not PRINT_TABLE:
        print('Elapsed time for all goal FPs search: ' + str(end - start))

    false_positive_rate_spread = np.array(false_positive_rate_spread)
    true_positive_rate_spread = np.array(true_positive_rate_spread)
    nrem_class_accuracies = np.array(nrem_class_accuracies)
    rem_class_accuracies = np.array(rem_class_accuracies)
    accuracies = np.array(accuracies)
    kappas = np.array(kappas)

    return false_positive_rate_spread, true_positive_rate_spread, nrem_class_accuracies, rem_class_accuracies, accuracies, kappas


def run_roc(method_key, feature_set, data_dict, train_test_dict, legend_text, plot_color):
    """
        Plots ROC curve for specified feature set and classifier

        Args:
            method_key (str): Key for classifier getting used
            feature_set (dict): Features to pass to classifier
            data_dict (dict): Contains all the subject data for classifiaction
            train_test_dict (dict): Contains training/testing subject splits for all trials
            legend_text (str): Label for legend
            plot_color (RGBA): color to plot

        """

    method = METHOD_DICT[method_key]  # Classifier to test
    params = []

    if verbose:
        print('Running trials...')

    output = []

    for run in range(0, NUM_REPS_TRAIN_TEST):  # Pre-builds dictionary to pass for training/testing

        train_set, test_set = train_test_dict[run]

        trial_dictionary = dict()
        trial_dictionary['run'] = run
        trial_dictionary['method'] = method
        trial_dictionary['method_key'] = method_key
        trial_dictionary['feature_set'] = feature_set
        trial_dictionary['data_dict'] = data_dict
        trial_dictionary['train_set'] = train_set
        trial_dictionary['test_set'] = test_set

        params.append(trial_dictionary)

        if run_flag == utilities.RUN_REM or run_flag == utilities.RUN_SW:
            output.append(parallel_roc(trial_dictionary))

    # TODO: Figure out why parallelization is causing problems
    # if run_flag == utilities.RUN_SW:
    #    output = pool.map(parallel_roc,params)

    if verbose:
        print('Looping over trials...')

    # Create false positive rate range to interpolate results over
    false_positive_spread = []

    for i in range(0, NUM_FALSE_POSITIVE_POINTS_PLOT):
        false_positive_spread.append((i + 1) / (NUM_FALSE_POSITIVE_POINTS_PLOT * 1.0))

    false_positive_spread = np.array(false_positive_spread)
    true_positive_spread = np.zeros(np.shape(false_positive_spread))

    # Average the results of all trials
    if run_flag == utilities.RUN_SW:

        avg_performance_at_interpolated_points = []

        for run in range(0, NUM_REPS_TRAIN_TEST):
            false_positive_rate = output[run][0]
            true_positive_rate = output[run][1]
            performance_at_interpolated_points = output[run][3]  # Interpolation points for tables in paper

            # Adds up performance across all true positive thresholds, to average over trials
            for interpolated_point_index in range(0, len(performance_at_interpolated_points)):
                if len(avg_performance_at_interpolated_points) <= interpolated_point_index:
                    performance_for_run = np.array(performance_at_interpolated_points[interpolated_point_index])
                    avg_performance_at_interpolated_points.append(performance_for_run)
                else:
                    performance_for_run = np.array(performance_at_interpolated_points[interpolated_point_index])
                    avg_performance_at_interpolated_points[interpolated_point_index] = \
                        avg_performance_at_interpolated_points[interpolated_point_index] + performance_for_run

            true_positive_rate_interpolated = np.interp(false_positive_spread, false_positive_rate, true_positive_rate)
            true_positive_spread = true_positive_spread + true_positive_rate_interpolated

        true_positive_spread = true_positive_spread / NUM_REPS_TRAIN_TEST

        # Insert (0,0) point for plotting curves
        false_positive_spread = np.insert(false_positive_spread, 0, 0)
        true_positive_spread = np.insert(true_positive_spread, 0, 0)

        false_positive_spread = np.array(false_positive_spread)
        true_positive_spread = np.array(true_positive_spread)

        plt.plot(false_positive_spread, true_positive_spread, label=legend_text, color=plot_color)  # Plot line for ROC

        if PRINT_TABLE:
            print('\hline ' + utilities.string_from_features(feature_set) + ' & ')
            for interpolated_point_index in range(0, len(performance_at_interpolated_points)):
                performance_metrics = avg_performance_at_interpolated_points[
                                          interpolated_point_index] / NUM_REPS_TRAIN_TEST
                line = ''

                if interpolated_point_index > 0:
                    line = ' & '

                for performance_item in performance_metrics[:-1]:
                    line = line + str(round(performance_item, 3)) + ' & '
                if interpolated_point_index == 0:
                    line = line + str(round(performance_metrics[-1], 3)) + ' \\\\'
                else:
                    line = line + ' \\\\'
                print(line)

    if run_flag == utilities.RUN_REM:

        nrem_class_accuracy_spread = np.zeros(np.shape(false_positive_spread))
        rem_class_accuracy_spread = np.zeros(np.shape(false_positive_spread))
        accuracy_spread = np.zeros(np.shape(false_positive_spread))
        kappa_spread = np.zeros(np.shape(false_positive_spread))

        for run in range(0, NUM_REPS_TRAIN_TEST):
            # Get performance for trial
            false_positive_rate = output[run][0]
            true_positive_rate = output[run][1]
            nrem_class_accuracy = output[run][2]
            rem_class_accuracy = output[run][3]
            accuracies = output[run][4]
            kappas = output[run][5]

            # Interpolate to match the desired spread
            true_positive_rate_interpolated = np.interp(false_positive_spread, false_positive_rate, true_positive_rate)
            nrem_accuracy_interpolated = np.interp(false_positive_spread, false_positive_rate, nrem_class_accuracy)
            rem_accuracy_interpolated = np.interp(false_positive_spread, false_positive_rate, rem_class_accuracy)
            accuracy_interpolated = np.interp(false_positive_spread, false_positive_rate, accuracies)
            kappa_interpolated = np.interp(false_positive_spread, false_positive_rate, kappas)

            # Add to cumulative totals for each value
            true_positive_spread = true_positive_spread + true_positive_rate_interpolated
            nrem_class_accuracy_spread = nrem_class_accuracy_spread + nrem_accuracy_interpolated
            rem_class_accuracy_spread = rem_class_accuracy_spread + rem_accuracy_interpolated
            accuracy_spread = accuracy_spread + accuracy_interpolated
            kappa_spread = kappa_spread + kappa_interpolated

        # Divide by number of trials to get average
        true_positive_spread = true_positive_spread / NUM_REPS_TRAIN_TEST
        nrem_class_accuracy_spread = nrem_class_accuracy_spread / NUM_REPS_TRAIN_TEST
        rem_class_accuracy_spread = rem_class_accuracy_spread / NUM_REPS_TRAIN_TEST
        accuracy_spread = accuracy_spread / NUM_REPS_TRAIN_TEST
        kappa_spread = kappa_spread / NUM_REPS_TRAIN_TEST

        # For tables, interpolate to find threshold where desired false positive rate is met
        nrem_accuracy_at_interpolated_point = np.interp(FALSE_POSITIVE_INTERPOLATION_POINT_REM_NREM_TABLES,
                                                        false_positive_spread, nrem_class_accuracy_spread)

        rem_accuracy_at_interpolated_point = np.interp(FALSE_POSITIVE_INTERPOLATION_POINT_REM_NREM_TABLES,
                                                       false_positive_spread, rem_class_accuracy_spread)

        index_of_best_accuracy = np.argmax(accuracy_spread)

        if PRINT_TABLE:
            print('\hline ' + utilities.string_from_features(feature_set) + ' & ')
            line = str(round(FALSE_POSITIVE_INTERPOLATION_POINT_REM_NREM_TABLES, 3)) + ' & ' \
                   + str(round(nrem_accuracy_at_interpolated_point, 3)) + ' & ' \
                   + str(round(rem_accuracy_at_interpolated_point, 3))
            line = line + ' & ' + str(round(accuracy_spread[index_of_best_accuracy], 3)) + ' & ' + \
                   str(round(kappa_spread[index_of_best_accuracy], 3))
            line = line + ' \\\\'
            print(line)

        # Insert(0,0) point for ROC curve
        false_positive_spread = np.insert(false_positive_spread, 0, 0)
        true_positive_spread = np.insert(true_positive_spread, 0, 0)
        nrem_class_accuracy_spread = np.insert(nrem_class_accuracy_spread, 0, 0)
        rem_class_accuracy_spread = np.insert(rem_class_accuracy_spread, 0, 0)

        false_positive_spread = np.array(false_positive_spread)
        true_positive_spread = np.array(true_positive_spread)

        tps_nrem = np.array(nrem_class_accuracy_spread)
        tps_rem = np.array(rem_class_accuracy_spread)

        # Plot line for ROC
        plt.plot(false_positive_spread, true_positive_spread, label=legend_text, color=plot_color)
        plt.plot(false_positive_spread, tps_nrem, color=plot_color, linestyle=':')
        plt.plot(false_positive_spread, tps_rem, color=plot_color, linestyle='--')


def make_method_roc(method_key):
    """
        Plots ROC curve for all feature sets given classifier

        Args:
            method_key (str): Key for classifier to plot

        """

    start = time.time()

    if verbose:
        print("Starting method ROC...")

    if PRINT_TABLE and run_flag == utilities.RUN_SW:
        print('\\begin{table}  \caption{' + method_key +
              ' Summary Statistics} \\begin{tabular}{l*{5}{c}} & Accuracy & Specificity & Sensitivity & $\kappa$ & AUC \\\\ ')

    if PRINT_TABLE and run_flag == utilities.RUN_REM:
        print('\\begin{table}  \caption{' + method_key +
              ' REM Summary Statistics} \\begin{tabular}{l*{5}{c}} & Wake Correct & NREM Correct & REM Correct & Best accuracy & $\kappa$ \\\\ ')

    # Loop over all feature sets
    for feature_set_index in range(0, len(feature_sets)):

        data_dict = utilities.build_data_dictionary(feature_sets[feature_set_index])  # Loads all data to dict

        run_roc(method_key, feature_sets[feature_set_index], data_dict, train_test_dict, cases[feature_set_index],
                colors[feature_set_index])  # Plots ROC curve for feature set

        end = time.time()

        if not PRINT_TABLE:
            print('Elapsed time: ' + str(end - start))

    if PRINT_TABLE and run_flag == utilities.RUN_SW:
        print('\end{tabular}  \label{tab:' + method_key[0:4] + 'params} \end{table}')

    if PRINT_TABLE and run_flag == utilities.RUN_REM:
        print('\end{tabular}  \label{tab:' + method_key[0:4] + '_rem_params} \end{table}')

    utilities.tidy_plot()
    font = font_manager.FontProperties(family='Arial', style='normal', size=14)

    if method_key == 'MLP':  # Add legend
        plt.legend(bbox_to_anchor=(1.0, 0.4), borderaxespad=0., prop=font)

    plt.xlabel('False positive rate', fontsize=16, fontname=font_name)
    plt.ylabel('True positive rate', fontsize=16, fontname=font_name)

    plt.title(method_key, fontsize=18, fontname=font_name, fontweight='bold')

    if run_flag == utilities.RUN_REM:
        type_string = '_rem_'
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 0.8])
    else:
        type_string = '_sw_'

    plt.savefig(method_key + '_' + str(NUM_REPS_TRAIN_TEST) + description + type_string + '_roc.png')
    plt.close()


def run_all(flag, trial_count):
    """
        Call to run all classifiers for either sleep/wake or wake/NREM/REM

        Args:
            flag (int): Type of classification to run (wake/sleep, or wake/NREM/REM)
            trial_count(int): How many times to repeat training and testing

        """

    global train_test_dict
    global run_flag
    global NUM_REPS_TRAIN_TEST
    global description

    run_flag = flag
    NUM_REPS_TRAIN_TEST = trial_count
    plt.ioff()
    description = 'output'

    pool = multiprocessing.Pool(processes=8)

    # Use a consistent train/test set across classifiers
    train_test_dict = utilities.make_train_test_dict(NUM_REPS_TRAIN_TEST)

    for method_key in METHOD_DICT.keys():
        if not PRINT_TABLE:
            print(method_key)
        make_method_roc(method_key)

    pool.close()
    pool.join()

    print('\a')


def run_one(method_key, flag, trial_count):
    """
        Call to run a single classifier for either sleep/wake or wake/NREM/REM

        Args:
            method_key (str): Key for classifier to use
            flag (int): Type of classification to run (wake/sleep, or wake/NREM/REM)
            trial_count(int): How many times to repeat training and testing

        """

    global train_test_dict
    global run_flag
    global NUM_REPS_TRAIN_TEST
    global description

    run_flag = flag
    NUM_REPS_TRAIN_TEST = trial_count
    plt.ioff()
    description = 'output'

    pool = multiprocessing.Pool(processes=8)

    # Use a consistent train/test set across classifiers
    train_test_dict = utilities.make_train_test_dict(NUM_REPS_TRAIN_TEST, 0.1)

    make_method_roc(method_key)

    pool.close()
    pool.join()


# Debugging: Prints subject performance
def check_subjects():
    method_key = 'MLP'
    global run_flag
    run_flag = utilities.RUN_SW
    feature_set = {'Motion': False, 'HR': True, 'Clock': False, 'Time': False, 'CircModel': False}
    export_all_subjects(feature_set, method_key)


# For sleep model/Kalman filter, saves classifier probabilities to file
def sleep_model_export():
    method_key = 'MLP'
    global run_flag
    run_flag = utilities.RUN_REM
    feature_set = {'Motion': True, 'HR': True, 'Clock': False, 'Time': False, 'CircModel': False}
    export_all_subjects(feature_set, method_key)


# For Kalman filter and debugging, train on all subjects but one; save probabilities for tested class:
def export_all_subjects(feature_set, method_key):
    data_dict = utilities.build_data_dictionary(feature_set)
    train_set = utilities.FULL_SET

    for ind in range(0, len(train_set)):
        subject_id = train_set[ind]
        if ind > 0:
            train_set_temp = train_set[0:ind]
            train_set_temp = train_set_temp + (train_set[ind + 1:])
        else:
            train_set_temp = train_set[1:]

        train_and_test_model(train_set_temp, [subject_id], method_key,
                             METHOD_DICT[method_key], feature_set, data_dict, True)


if __name__ == '__main__':
    # check_subjects()
    sleep_model_export()
