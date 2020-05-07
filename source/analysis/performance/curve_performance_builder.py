import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, cohen_kappa_score

from source.analysis.performance.curve_performance import ROCPerformance, PrecisionRecallPerformance
from source.analysis.performance.epoch_performance import ThreeClassPerformance
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.sleep_label import SleepWakeLabel
from source.analysis.setup.sleep_labeler import SleepLabeler


class CurvePerformanceBuilder(object):
    NUMBER_OF_INTERPOLATION_POINT = 100

    @staticmethod
    def get_axes_bins():
        x_axis = []

        for i in range(0, CurvePerformanceBuilder.NUMBER_OF_INTERPOLATION_POINT):
            x_axis.append((i + 1) / (CurvePerformanceBuilder.NUMBER_OF_INTERPOLATION_POINT * 1.0))

        x_axis = np.array(x_axis)
        y_axis = np.zeros(np.shape(x_axis))

        return x_axis, y_axis

    @staticmethod
    def build_roc_from_raw(raw_performances: [RawPerformance], positive_class):
        false_positive_spread, true_positive_spread = CurvePerformanceBuilder.get_axes_bins()
        count = 0

        for raw_performance in raw_performances:
            true_labels = SleepLabeler.label_one_vs_rest(raw_performance.true_labels, positive_class)

            false_positive_rates, true_positive_rates, thresholds = roc_curve(
                true_labels,
                raw_performance.class_probabilities[:, positive_class],
                pos_label=SleepWakeLabel.sleep.value,
                drop_intermediate=False)
            count = count + 1
            true_positive_spread += np.interp(false_positive_spread, false_positive_rates,
                                              true_positive_rates)

        true_positive_spread = true_positive_spread / count

        false_positive_spread = np.insert(false_positive_spread, 0, 0)
        true_positive_spread = np.insert(true_positive_spread, 0, 0)

        return ROCPerformance(false_positive_rates=false_positive_spread, true_positive_rates=true_positive_spread)

    @staticmethod
    def build_precision_recall_from_raw(raw_performances: [RawPerformance]):
        recall_spread, precision_spread = CurvePerformanceBuilder.get_axes_bins()
        count = 0

        for raw_performance in raw_performances:
            count = count + 1
            precisions, recalls, thresholds = precision_recall_curve(raw_performance.true_labels,
                                                                     raw_performance.class_probabilities[:, 0],
                                                                     pos_label=SleepWakeLabel.wake.value)
            precision_spread += np.interp(recall_spread, np.flip(recalls), np.flip(precisions))

        precision_spread = precision_spread / count

        recall_spread = np.array(np.insert(recall_spread, 0, 0))
        precision_spread = np.array(np.insert(precision_spread, 0, 1))

        return PrecisionRecallPerformance(recalls=recall_spread, precisions=precision_spread)

    @staticmethod
    def build_three_class_roc_with_binary_search(raw_performances: [RawPerformance]):
        ''' This function runs two binary searches-- first, to match a given wake accuracy value,
        and second, the try to find a threshold that balances the NREM and REM class accuracies '''
        three_class_performances = []

        number_of_wake_scored_as_sleep_bins = 20  # Bin size for the x-dimension of this plot
        false_positive_buffer = 0.001  # How close to the target wake false positive rate we need to be before stopping
        max_attempts_binary_search_wake = 50  # Max number of times the search will try to find a given wake accuracy
        rem_nrem_accuracy_tolerance = 1e-2  # How close the NREM and REM class accuracies need to be before stopping
        max_attempts_binary_search_rem_nrem = 15  # Max times the search will try to balance REM and NREM accuracies
        wake_scored_as_sleep_interpolation_point = 0.4  # Point to report values in the table for
        goal_fraction_wake_scored_as_sleep_spread = []  # Wake false positive rate spread (x-axis of plot)

        #  Initialize holders for wake, NREM, REM accuracies
        for i in range(0, number_of_wake_scored_as_sleep_bins):
            goal_fraction_wake_scored_as_sleep_spread.append(i / (number_of_wake_scored_as_sleep_bins * 1.0))

        goal_fraction_wake_scored_as_sleep_spread = np.array(goal_fraction_wake_scored_as_sleep_spread)
        cumulative_nrem_accuracies = np.zeros(np.shape(goal_fraction_wake_scored_as_sleep_spread))
        cumulative_rem_accuracies = np.zeros(np.shape(goal_fraction_wake_scored_as_sleep_spread))
        cumulative_accuracies = np.zeros(np.shape(goal_fraction_wake_scored_as_sleep_spread))

        cumulative_counter = 0

        # Loop over all training/testing splits
        for raw_performance in raw_performances:

            true_labels = raw_performance.true_labels
            class_probabilities = raw_performance.class_probabilities

            wake_scored_as_sleep_spread = []
            sleep_accuracy_spread = []
            accuracies = []
            kappas = []
            nrem_class_accuracies = []
            rem_class_accuracies = []

            true_wake_indices = np.where(true_labels == 0)[0]
            true_nrem_indices = np.where(true_labels == 1)[0]
            true_rem_indices = np.where(true_labels == 2)[0]

            #  Try to find a threshold that matches a target fraction wake scored as sleep
            for goal_fraction_wake_scored_as_sleep in goal_fraction_wake_scored_as_sleep_spread:

                fraction_wake_scored_as_sleep = -1
                binary_search_counter = 0

                # While we haven't found the target wake false positive rate
                # (and haven't exceeded the number of allowable searches), keep searching:
                while (fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer
                       or fraction_wake_scored_as_sleep >= goal_fraction_wake_scored_as_sleep + false_positive_buffer) \
                        and binary_search_counter < max_attempts_binary_search_wake:

                    # If this is the first iteration on the binary search, initialize.
                    if binary_search_counter == 0:
                        threshold_for_sleep = 0.5
                        threshold_delta = 0.25
                    else:
                        if fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer:
                            threshold_for_sleep = threshold_for_sleep - threshold_delta
                            threshold_delta = threshold_delta / 2
                        if fraction_wake_scored_as_sleep >= goal_fraction_wake_scored_as_sleep + false_positive_buffer:
                            threshold_for_sleep = threshold_for_sleep + threshold_delta
                            threshold_delta = threshold_delta / 2

                    if goal_fraction_wake_scored_as_sleep == 1:  # Edge cases
                        threshold_for_sleep = 0.0
                    if goal_fraction_wake_scored_as_sleep == 0:
                        threshold_for_sleep = 1.0

                    predicted_sleep_indices = np.where(1 - np.array(class_probabilities[:, 0]) >= threshold_for_sleep)[0]

                    predicted_labels = np.zeros(np.shape(true_labels))
                    predicted_labels[predicted_sleep_indices] = 1
                    predicted_labels_at_true_wake_indices = predicted_labels[true_wake_indices]

                    number_wake_correct = len(true_wake_indices) - np.count_nonzero(
                        predicted_labels_at_true_wake_indices)
                    fraction_wake_correct = number_wake_correct / (len(true_wake_indices) * 1.0)
                    fraction_wake_scored_as_sleep = 1.0 - fraction_wake_correct

                    binary_search_counter = binary_search_counter + 1

                # Next, try to find a threshold that balances the REM and NREM class accuracies
                if binary_search_counter < max_attempts_binary_search_wake:

                    smallest_accuracy_difference = 2
                    sleep_accuracy = 0
                    rem_accuracy = 0
                    nrem_accuracy = 0
                    best_accuracy = -1
                    kappa_at_best_accuracy = -1

                    count_thresh = 0
                    threshold_for_rem = 0.5
                    threshold_delta_rem = 0.5

                    # While we're outside the tolerance window for equalizing REM/NREM accuracies and
                    # we haven't exceed the maximum number of search attempts, keep hunting
                    while count_thresh < max_attempts_binary_search_rem_nrem and \
                            smallest_accuracy_difference > rem_nrem_accuracy_tolerance:

                        count_thresh = count_thresh + 1

                        for predicted_sleep_index in range(len(predicted_sleep_indices)):
                            predicted_sleep_epoch = predicted_sleep_indices[predicted_sleep_index]

                            # Apply the threshold to split REM and NREM sleep
                            if class_probabilities[predicted_sleep_epoch, 2] > threshold_for_rem:
                                predicted_labels[predicted_sleep_epoch] = 2  # Set to REM sleep
                            else:
                                predicted_labels[predicted_sleep_epoch] = 1  # Set to NREM sleep

                        accuracy = accuracy_score(predicted_labels, true_labels)
                        kappa = cohen_kappa_score(predicted_labels, true_labels)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            kappa_at_best_accuracy = kappa

                        # Assess accuracy at the current threshold
                        predicted_nrem_indices = np.where(predicted_labels == 1)[0]
                        predicted_rem_indices = np.where(predicted_labels == 2)[0]

                        correct_nrem_indices = np.intersect1d(predicted_nrem_indices, true_nrem_indices)
                        correct_rem_indices = np.intersect1d(predicted_rem_indices, true_rem_indices)

                        nrem_accuracy = len(correct_nrem_indices) / (1.0 * len(true_nrem_indices))

                        if len(true_rem_indices) > 0:
                            rem_accuracy = len(correct_rem_indices) / (1.0 * len(true_rem_indices))
                        else:
                            rem_accuracy = 0

                        # Update current accuracy difference and adjust the threshold for REM sleep
                        sleep_accuracy = (len(correct_nrem_indices) + len(correct_rem_indices)) / (
                                1.0 * len(true_nrem_indices) + 1.0 * len(true_rem_indices))

                        smallest_accuracy_difference = np.abs(nrem_accuracy - rem_accuracy)

                        if rem_accuracy < nrem_accuracy:
                            threshold_for_rem = threshold_for_rem - threshold_delta_rem / 2.0
                        else:
                            threshold_for_rem = threshold_for_rem + threshold_delta_rem / 2.0

                        threshold_delta_rem = threshold_delta_rem / 2.0

                    wake_scored_as_sleep_spread.append(fraction_wake_scored_as_sleep)
                    sleep_accuracy_spread.append(sleep_accuracy)
                    nrem_class_accuracies.append(nrem_accuracy)
                    rem_class_accuracies.append(rem_accuracy)
                    accuracies.append(best_accuracy)
                    kappas.append(kappa_at_best_accuracy)

            wake_scored_as_sleep_spread = np.array(wake_scored_as_sleep_spread)
            sleep_accuracy_spread = np.array(sleep_accuracy_spread)
            nrem_class_accuracies = np.array(nrem_class_accuracies)
            rem_class_accuracies = np.array(rem_class_accuracies)

            wake_scored_as_sleep_spread = np.insert(wake_scored_as_sleep_spread, 0, 0)
            sleep_accuracy_spread = np.insert(sleep_accuracy_spread, 0, 0)
            nrem_class_accuracies = np.insert(nrem_class_accuracies, 0, 0)
            rem_class_accuracies = np.insert(rem_class_accuracies, 0, 0)

            index_of_best_accuracy = np.argmax(accuracies)

            accuracy = accuracies[index_of_best_accuracy]
            kappa = kappas[index_of_best_accuracy]

            # Interpolate and add the vectors to our running totals
            cumulative_accuracies = cumulative_accuracies + np.interp(goal_fraction_wake_scored_as_sleep_spread,
                                                                      wake_scored_as_sleep_spread,
                                                                      sleep_accuracy_spread)

            cumulative_nrem_accuracies = cumulative_nrem_accuracies + np.interp(
                goal_fraction_wake_scored_as_sleep_spread,
                wake_scored_as_sleep_spread,
                nrem_class_accuracies)

            cumulative_rem_accuracies = cumulative_rem_accuracies + np.interp(goal_fraction_wake_scored_as_sleep_spread,
                                                                              wake_scored_as_sleep_spread,
                                                                              rem_class_accuracies)

            cumulative_counter = cumulative_counter + 1

            rem_correct = np.interp(wake_scored_as_sleep_interpolation_point,
                                    wake_scored_as_sleep_spread,
                                    rem_class_accuracies)
            nrem_correct = np.interp(wake_scored_as_sleep_interpolation_point,
                                     wake_scored_as_sleep_spread,
                                     nrem_class_accuracies)

            three_class_performance = ThreeClassPerformance(accuracy=accuracy,
                                                            wake_correct=1 - wake_scored_as_sleep_interpolation_point,
                                                            rem_correct=rem_correct,
                                                            nrem_correct=nrem_correct,
                                                            kappa=kappa)
            three_class_performances.append(three_class_performance)

        # Average over all the train/test splits
        cumulative_accuracies = cumulative_accuracies / cumulative_counter
        cumulative_nrem_accuracies = cumulative_nrem_accuracies / cumulative_counter
        cumulative_rem_accuracies = cumulative_rem_accuracies / cumulative_counter

        sleep_wake_roc_performance = ROCPerformance(false_positive_rates=goal_fraction_wake_scored_as_sleep_spread,
                                                    true_positive_rates=cumulative_accuracies)

        rem_roc_performance = ROCPerformance(false_positive_rates=goal_fraction_wake_scored_as_sleep_spread,
                                             true_positive_rates=cumulative_rem_accuracies)

        nrem_roc_performance = ROCPerformance(false_positive_rates=goal_fraction_wake_scored_as_sleep_spread,
                                              true_positive_rates=cumulative_nrem_accuracies)

        return sleep_wake_roc_performance, rem_roc_performance, nrem_roc_performance, three_class_performances
