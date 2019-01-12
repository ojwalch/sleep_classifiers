import sys
import math
import csv
import numpy as np

from xml.dom import minidom
import pyedflib
import glob

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, cohen_kappa_score, recall_score, precision_score, \
    accuracy_score

import make_features
import classify_sleep
import utilities

verbose = False
metadata_dict = {}
WANT_LOAD_MODEL = True


# Generate MESA cohort summary statistics
def load_metadata():
    ref_dict = dict()
    ref_dict['ahi'] = 'ahiu35'
    ref_dict['age'] = 'sleepage5c'
    ref_dict['gender'] = 'gender1' # 0 Female, 1 Male
    ref_dict['tst'] = 'slpprdp5'
    ref_dict['tib'] = 'time_bed5'
    ref_dict['waso'] = 'waso5'
    ref_dict['slp_eff'] = 'slp_eff5'
    ref_dict['time_rem'] = 'timerem5'
    ref_dict['time_n1'] = 'timest15'
    ref_dict['time_n2'] = 'timest25'
    ref_dict['time_n34'] = 'timest345'
    
    with open('../mesa/mesa-sleep-dataset-0.3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        is_first_row = True
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                count = 0
                for col in row:
                    if col in ref_dict.values():
                        key = ref_dict.keys()[ref_dict.values().index(col)]
                        ref_dict[key] = count  # Replace with index number where data is
                    count = count + 1
            else:
                subject_dict = {}
                for key in ref_dict:
                    val = row[ref_dict[key]]
                    if len(val) > 0:
                        subject_dict[key] = float(val)
                    else:
                        subject_dict[key] = np.nan
                metadata_dict[int(row[0])] = subject_dict


# Print table with MESA summary statistics
def print_metadata(subject_inds):
    
    ahis = []
    ages = []
    genders = []
    tst = []
    tib = []
    waso = []
    slp_eff = []
    time_rem = []
    time_nrem = []
    
    for subject_index in subject_inds:                  # Get all metadata for subjects
        subject = int(all_files[subject_index][-8:-4])
        subject_dict = metadata_dict[subject]
        ahis.append(subject_dict['ahi'])
        ages.append(subject_dict['age'])
        genders.append(subject_dict['gender'])
        tst.append(subject_dict['tst'])
        tib.append(subject_dict['tib'])
        waso.append(subject_dict['waso'])
        slp_eff.append(subject_dict['slp_eff'])
        time_rem.append(subject_dict['time_rem'])
        time_nrem.append(float(subject_dict['time_n1']) + float(subject_dict['time_n2']) + float(subject_dict['time_n34']))

    ahis = np.array(ahis)
    ages = np.array(ages)
    genders = np.array(genders)
    tst = np.array(tst)
    tib = np.array(tib)
    waso = np.array(waso)
    slp_eff = np.array(slp_eff)

    print('N women: ' + str(len(genders) - np.count_nonzero(genders)))
    
    latex = True
    if(latex):
        print('\\begin{table} \caption{Sleep summary statistics - MESA subcohort}  \label{tab:mesa} \small  \\begin{tabularx}{\columnwidth}{X | X | X  }\hline Parameter & Mean (SD) & Range \\\\ \hline')
    else:
        print('Parameter, Mean (SD), Range')

    print(utilities.data_to_line('Age (years)',ages,latex))
    print(utilities.data_to_line('TST (minutes)',tst,latex))
    print(utilities.data_to_line('TIB (minutes)',tib,latex))
    print(utilities.data_to_line('WASO (minutes)',waso,latex))
    print(utilities.data_to_line('SE (\%)',slp_eff,latex))
    print(utilities.data_to_line('AHI',ahis,latex))

    if(latex):
        print('\end{tabularx} \end{table}')


def get_features_and_labels(time, heart_rate, activity, stages, circ_model, feature_set):
    """
         Convert MESA data to features and labels compatible with Apple Watch data

         Args:
             time (np.array): Time vector
             heart_rate (np.array) : Heart rate data from MESA PSG EKG
             activity (np.array) : Activity counts from MESA actigraph
             stages (np.array) : PSG Sleep stages
             circ_model (np.array) : Circadian model prediction from ODE integration
             feature_set (dict) : Feature sets to test

         Returns:
            dict : Maps run count to training and testing set
         """

    window_size = make_features.WINDOW_SIZE
    dt_scores = make_features.DT_SCORES
    dt_motion = make_features.DT_MOTION
    dt_hr = make_features.DT_HR

    true_labels = []
    features = []
    
    duration = int(math.floor(time[-1] - time[0]))
    start_time = time[0]
    end_time = time[-1]

    if verbose:
        print('Duration : ' + str(duration))
    
    if feature_set['HR']:
        hr = np.transpose(np.vstack((np.transpose(time), np.transpose(heart_rate))))
        hr = make_features.process_hr(hr, start_time, end_time)

    invalid_count = 0

    # Make features following the conventions of make_features.py
    for i in range(window_size*dt_scores/2, duration - window_size*dt_scores/2, dt_scores):
        stages_in_range = np.array(stages[i:(i+dt_scores) - 1])

        if len(stages_in_range) > 0:
            valid_heartrate = True
            valid_circ_model = True
            
            sample_begin = int(start_time + i - window_size*dt_scores/2)
            sample_end = int(start_time + i + window_size*dt_scores/2)
            
            motion = np.transpose(np.vstack((np.transpose(time), np.transpose(activity))))
            motion_epoch = make_features.get_motion_feature(range(sample_begin, sample_end, dt_motion), motion)
            feature = np.array([])

            if feature_set['Motion']:
                feature = np.append(feature, motion_epoch)

            if feature_set['HR']:
                hr_epoch = make_features.get_hr_feature(range(sample_begin, sample_end, dt_hr), hr)

                if np.count_nonzero(hr_epoch) != len(hr_epoch):
                    valid_heartrate = False
                    invalid_count = invalid_count + 1
                
                feature = np.append(feature, hr_epoch)

            if feature_set['Clock']:
                clock_epoch = make_features.get_clock_feature(i)
                feature = np.append(feature, np.array(clock_epoch))

            if feature_set['CircModel']:
                if np.shape(circ_model)[0] > 2:
                    cm_feature = make_features.get_circ_model_feature(i, circ_model)
                    if ~np.isnan(np.sum(cm_feature)):
                        feature = np.append(feature, cm_feature)
                    else:
                        valid_circ_model = False
                        if verbose:
                            print('Invalid circadian output...')
                else:
                    valid_circ_model = False
                    if verbose:
                        print('Invalid circadian output...')

            if valid_heartrate and valid_circ_model:
                true_labels.append(stages_in_range[0])
                features.append(feature)

    if len(features) > 0 and verbose:
        print(features[0])
        print(features[-1])
        print('Subject has ' + str(invalid_count) + ' invalid epochs')
    
    return features, true_labels


def get_data_for_index(index, run_flag):
    """
         Load raw data for subject index

         Args:
            index (int) :  Subject index
            run_flag (int) : wake/sleep vs wake/NREM/REM
         Returns:
            time, heart rate, activity, clock model output, and PSG labels
         """
    if run_flag == utilities.RUN_REM:
        stage_to_num = {'Stage 4 sleep|4': 1, 'Stage 3 sleep|3': 1, 'Stage 2 sleep|2': 1, 'Stage 1 sleep|1': 1,
                        'Wake|0': 0, 'REM sleep|5': 2}
    else:
        stage_to_num = {'Stage 4 sleep|4': 1, 'Stage 3 sleep|3': 1, 'Stage 2 sleep|2': 1, 'Stage 1 sleep|1': 1,
                        'Wake|0': 0, 'REM sleep|5': 1}

    file_id = all_files[index][-8:-4]

    if verbose:
        print('Running subject: ' + file_id)

    xml_document = minidom.parse('../mesa/polysomnography/annotations-events-nsrr/mesa-sleep-' + file_id + '-nsrr.xml')
    list_of_scored_events = xml_document.getElementsByTagName('ScoredEvent')

    stage_data = []
    for scored_event in list_of_scored_events:  # 3 is stage, 5 is start, 7 is duration
        duration = scored_event.childNodes[7].childNodes[0].nodeValue
        start = scored_event.childNodes[5].childNodes[0].nodeValue
        stage = scored_event.childNodes[3].childNodes[0].nodeValue

        if stage in stage_to_num:
            # # For debugging: print(str(stage) + ' ' + str(start) + ' ' + str(duration))
            stage_data.append([stage_to_num[stage], float(start), float(duration)])

    edf_file = pyedflib.EdfReader('../mesa/polysomnography/edfs/mesa-sleep-' + file_id + '.edf')
    signal_labels = edf_file.getSignalLabels()

    # # For debugging: print(signal_labels)

    hr_column = len(signal_labels) - 2
    sample_frequencies = edf_file.getSampleFrequencies()

    heart_rate = edf_file.readSignal(hr_column)
    sf = sample_frequencies[hr_column]
    
    time_hr = np.array(range(0, len(heart_rate)))  # Get timestamps for heart rate data
    time_hr = time_hr/sf

    stages = []
    for staged_window in stage_data[:-1]:  # Ignore last PSG overflow entry: it's long & doesn't have valid HR
        elapsed_time_counter = 0
        stage_value = staged_window[0]
        duration = staged_window[2]
        while elapsed_time_counter < duration:
            stages.append(stage_value)
            elapsed_time_counter = elapsed_time_counter + 1

    time = range(0, len(stages))  # Units are seconds
    heart_rate = np.interp(time, time_hr, heart_rate)

    line_align = -1                     # Find alignment line between PSG and actigraphy
    with open('../mesa/overlap/mesa-actigraphy-psg-overlap.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            if int(row[0]) == int(file_id):
                line_align = int(row[1])

    activity = []
    elapsed_time_counter = 0
    
    if line_align == -1:                # If there was not alignment found
        return -1, -1, -1, -1, -1

    with open('../mesa/actigraphy/mesa-sleep-' + file_id + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            if int(row[1]) > line_align:
                if(row[4] == ''):
                    activity.append([elapsed_time_counter, 0.0])
                else:
                    activity.append([elapsed_time_counter, float(row[4])])
                elapsed_time_counter = elapsed_time_counter + 30

    circ_model = np.genfromtxt('../mesa/clock_proxy/' + file_id + '_clock_proxy.out', delimiter=',')

    activity = np.array(activity)
    activity_interpolated = np.interp(time, activity[:, 0], activity[:, 1])

    return time, heart_rate, activity_interpolated, circ_model, stages


def make_roc_mesa(run_flag):
    """
         Make ROC curve for MESA and save image to file

         Args:
            run_flag (int) : wake/sleep vs wake/NREM/REM
         """

    rand_perm = np.random.permutation(len(all_files))

    print('Total available files: ' + str(len(rand_perm)))

    if not WANT_LOAD_MODEL:
        train_indices = rand_perm[0:100]
        test_indices = rand_perm[101:]
    else:
        test_indices = rand_perm

    print_metadata(test_indices)

    # Run one trial of MLP classifier on Apple Watch data to save the trained model to file
    method_key = 'MLP'
    classify_sleep.run_one(method_key, run_flag, 1)

    if classify_sleep.PRINT_TABLE and run_flag == utilities.RUN_SW:
        print('\\begin{table}  \caption{' + method_key + ' MESA Performance} \\begin{tabular}{l*{5}{c}} & Accuracy & Specificity & Sensitivity & $\kappa$ & AUC \\\\ ')
    
    if classify_sleep.PRINT_TABLE and run_flag == utilities.RUN_REM:
        print('\\begin{table}  \caption{' + method_key + ' REM MESA Performance} \\begin{tabular}{l*{5}{c}} & Wake Correct & NREM Correct & REM Correct & Best accuracy & $\kappa$ \\\\ ')

    data_dictionary = {}

    if not WANT_LOAD_MODEL:
        for train_index in train_indices:
            data_dictionary[train_index] = get_data_for_index(train_index, run_flag)

    for test_index in test_indices:
        data_dictionary[test_index] = get_data_for_index(test_index, run_flag)

    for feature_set_index in range(0, len(classify_sleep.feature_sets)):
        feature_set = classify_sleep.feature_sets[feature_set_index]

        if not WANT_LOAD_MODEL:  # If we *don't* want to test on Apple Watch data
            training_set_features = []
            training_set_true_labels = []
            for train_index in train_indices:
                time, heart_rate, activity, circ_model, stages = data_dictionary[train_index]
                if time != -1:
                    training_set_features_temp, training_set_labels_temp = get_features_and_labels(time, heart_rate, activity, stages, circ_model, feature_set)
                    training_set_features.extend(training_set_features_temp)
                    training_set_true_labels.extend(training_set_labels_temp)

            classifier = classify_sleep.METHOD_DICT[method_key]
            classifier.fit(training_set_features, training_set_true_labels)
            classifier_abbrev = str(classifier)[0:4]
            save_name = 'trained_models/' + classifier_abbrev + utilities.string_from_features(feature_set) \
                        + '_trained_modelMESA.npy'
            np.save(save_name, classifier)

        testing_set_features = []
        testing_set_labels = []

        for test_index in test_indices:
            time, heart_rate, activity, circ_model, stages = data_dictionary[test_index]

            if time != -1:
                testing_set_features_temp, testing_set_labels_temp = get_features_and_labels(time, heart_rate,
                                                                                             activity, stages,
                                                                                             circ_model, feature_set)
                testing_set_features.extend(testing_set_features_temp)
                testing_set_labels.extend(testing_set_labels_temp)

        if WANT_LOAD_MODEL:
            load_name = str(classify_sleep.METHOD_DICT[method_key])
            load_name = load_name[0:4]
            if not classify_sleep.PRINT_TABLE:
                print('Loading trained_models/' + load_name + utilities.string_from_features(feature_set)
                      + '_trained_model.npy')
            classifier = np.load('trained_models/' + load_name + utilities.string_from_features(feature_set) + '_trained_model.npy')
            if method_key == 'Random Forest':
                classifier = classifier[0]
            else:
                classifier = classifier.item()

        class_probabilities = classifier.predict_proba(testing_set_features)     # Get class probabilities
        testing_set_labels = np.array(testing_set_labels)

        if run_flag == utilities.RUN_SW:
            false_positive_rates, true_positive_rates, thresholds = \
                roc_curve(testing_set_labels, classifier.predict_proba(testing_set_features)[:, 1])
            performance = utilities.thresh_interpolation(false_positive_rates, true_positive_rates, thresholds,
                                                         class_probabilities, testing_set_labels)

            if classify_sleep.PRINT_TABLE:
                print('\hline ' + utilities.string_from_features(feature_set) + ' & ')
                for p_count in range(0,len(performance)):
                    
                    stats = performance[p_count]
                    line = ''
                    
                    if p_count > 0:
                        line = ' & '
                    
                    for stat in stats[:-1]:
                        line = line + str(round(stat,3)) + ' & '
                    if p_count == 0:
                        line = line + str(round(stats[-1],3)) + ' \\\\'
                    else:
                        line = line + ' \\\\'
                    print(line)

            plt.plot(false_positive_rates, true_positive_rates, color=classify_sleep.colors[feature_set_index],
                     label=classify_sleep.cases[feature_set_index])
            plt.xlabel('False positive rate', fontsize=16)
            plt.ylabel('True positive rate', fontsize=16)
            utilities.tidy_plot()
            plt.legend(bbox_to_anchor=(1.0, 0.4), borderaxespad=0., prop={'size': 10})
            plt.savefig('figure_mesa_roc.png')

        else:
            false_positive_rates, true_positive_rates, nrem_accuracies, rem_accuracies, accuracies, kappas = \
                classify_sleep.roc_curve_rem(testing_set_labels, classifier.predict_proba(testing_set_features))
            
            false_positive_rate_spread = []
            for i in range(0, classify_sleep.NUM_FALSE_POSITIVE_POINTS_PLOT):
                false_positive_rate_spread.append((i+1)/(classify_sleep.NUM_FALSE_POSITIVE_POINTS_PLOT*1.0))

            false_positive_rate_spread = np.array(false_positive_rate_spread)

            true_positive_rate_spread = np.interp(false_positive_rate_spread, false_positive_rates, true_positive_rates)
            nrem_accuracy_spread = np.interp(false_positive_rate_spread, false_positive_rates, nrem_accuracies)
            rem_accuracy_spread = np.interp(false_positive_rate_spread, false_positive_rates, rem_accuracies)
            acc_spread = np.interp(false_positive_rate_spread, false_positive_rates, accuracies)
            kappa_spread = np.interp(false_positive_rate_spread, false_positive_rates, kappas)

            false_positive_rate_spread = np.insert(false_positive_rate_spread, 0, 0)
            true_positive_rate_spread = np.insert(true_positive_rate_spread, 0, 0)
            nrem_accuracy_spread = np.insert(nrem_accuracy_spread, 0, 0)
            rem_accuracy_spread = np.insert(rem_accuracy_spread, 0, 0)

            false_positive_rate_spread = np.array(false_positive_rate_spread)
            true_positive_rate_spread = np.array(true_positive_rate_spread)

            nrem_accuracy_spread = np.array(nrem_accuracy_spread)
            rem_accuracy_spread = np.array(rem_accuracy_spread)

            false_positive_interpolation_point = classify_sleep.FALSE_POSITIVE_INTERPOLATION_POINT_REM_NREM_TABLES
            rem_accuracy_interpolation_point = np.interp(false_positive_interpolation_point, false_positive_rate_spread,
                                                         rem_accuracy_spread)
            nrem_accuracy_interpolation_point = np.interp(false_positive_interpolation_point, false_positive_rate_spread,
                                                          nrem_accuracy_spread)

            index_of_best_accuracy = np.argmax(acc_spread)

            if classify_sleep.PRINT_TABLE:
                print('\hline ' + utilities.string_from_features(feature_set) + ' & ')
                line = str(round(false_positive_interpolation_point, 3)) + ' & ' \
                       + str(round(nrem_accuracy_interpolation_point, 3)) + ' & ' \
                       + str(round(rem_accuracy_interpolation_point, 3))
                line = line + ' & ' + str(round(acc_spread[index_of_best_accuracy], 3)) \
                       + ' & ' + str(round(kappa_spread[index_of_best_accuracy], 3))

                line = line + ' \\\\'
                print(line)

            plt.plot(false_positive_rate_spread, true_positive_rate_spread,
                     label=classify_sleep.cases[feature_set_index], color=classify_sleep.colors[feature_set_index])
            plt.plot(false_positive_rate_spread, nrem_accuracy_spread,
                     color=classify_sleep.colors[feature_set_index], linestyle=':')
            plt.plot(false_positive_rate_spread, rem_accuracy_spread,
                     color=classify_sleep.colors[feature_set_index], linestyle='--')
            plt.xlabel('False positive rate', fontsize=16)
            plt.ylabel('True positive rate', fontsize=16)
            utilities.tidy_plot()
            plt.legend(bbox_to_anchor=(1.0, 0.4), borderaxespad=0., prop={'size': 10})

            plt.savefig('figure_mesa_roc_rem.png')
                
    if classify_sleep.PRINT_TABLE and run_flag == utilities.RUN_SW:
        print('\end{tabular}  \label{tab:' + method_key[0:4] + 'mesa} \end{table}')
    
    if classify_sleep.PRINT_TABLE and run_flag == utilities.RUN_REM:
        print('\end{tabular}  \label{tab:' + method_key[0:4] + '_rem_mesa} \end{table}')


all_files = glob.glob("../mesa/polysomnography/edfs/*edf")

load_metadata()
