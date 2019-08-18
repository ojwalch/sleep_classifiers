import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.performance.performance_builder import PerformanceBuilder
from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.performance.sleep_metrics_calculator import SleepMetricsCalculator
from source.analysis.setup.feature_set_service import FeatureSetService
from source.constants import Constants


class PerformancePlotBuilder(object):
    @staticmethod
    def make_histogram_with_thresholds(classifier_summary: ClassifierSummary):

        for feature_set in classifier_summary.performance_dictionary:
            raw_performances = classifier_summary.performance_dictionary[feature_set]

            number_of_thresholds = 4
            number_of_subjects = len(raw_performances)
            fig, ax = plt.subplots(nrows=number_of_thresholds, ncols=2, figsize=(8, 8), sharex=True, sharey=True)

            all_accuracies = np.zeros((number_of_subjects, number_of_thresholds))
            all_wake_correct_fractions = np.zeros((number_of_subjects, number_of_thresholds))

            for subject_index in range(number_of_subjects):
                true_positive_thresholds, performance_summary = PerformanceSummarizer.summarize_thresholds(
                    [raw_performances[subject_index]])

                for threshold_index in range(number_of_thresholds):
                    all_accuracies[subject_index, threshold_index] = performance_summary[threshold_index].accuracy
                    all_wake_correct_fractions[subject_index, threshold_index] = performance_summary[
                        threshold_index].wake_correct

            dt = 0.02
            row_count = 0
            font_size = 16
            font_name = 'Arial'

            for row in ax:
                row[0].hist(all_accuracies[:, row_count].tolist(),
                            bins=np.arange(0, 1 + dt, dt),
                            color="skyblue",
                            ec="skyblue")
                row[1].hist(all_wake_correct_fractions[:, row_count].tolist(),
                            bins=np.arange(0, 1 + dt, dt),
                            color="lightsalmon",
                            ec="lightsalmon")

                if row_count == number_of_thresholds - 1:
                    row[0].set_xlabel('Accuracy', fontsize=font_size, fontname=font_name)
                    row[1].set_xlabel('Wake correct', fontsize=font_size, fontname=font_name)
                    row[0].set_ylabel('Count', fontsize=font_size, fontname=font_name)
                row[0].set_xlim((0, 1))
                row[1].set_xlim((0, 1))

                row_count = row_count + 1

            file_save_name = str(Constants.FIGURE_FILE_PATH) + '/' + FeatureSetService.get_label(feature_set) + '_' \
                             + classifier_summary.attributed_classifier.name + '_histograms_with_thresholds.png'

            plt.tight_layout()

            plt.savefig(file_save_name, dpi=300)
            plt.close()

            image = Image.open(file_save_name)
            width, height = image.size
            scale_factor = 0.3
            new_image = Image.new('RGB', (int((1 + scale_factor) * width), height), "white")
            new_image.paste(image, (int(scale_factor * width), 0))
            draw = ImageDraw.Draw(new_image)
            font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 75)

            draw.text((int(scale_factor * width / 3), int((height * 0.9) * 0.125)), "TPR = 0.8", (0, 0, 0), font=font)
            draw.text((int(scale_factor * width / 3), int(height * 0.9 * 0.375)), "TPR = 0.9", (0, 0, 0), font=font)
            draw.text((int(scale_factor * width / 3), int(height * 0.9 * 0.625)), "TPR = 0.93", (0, 0, 0), font=font)
            draw.text((int(scale_factor * width / 3), int(height * 0.9 * 0.875)), "TPR = 0.95", (0, 0, 0), font=font)

            new_image.save(str(Constants.FIGURE_FILE_PATH) + '/' + 'figure_threshold_histogram.png')

    @staticmethod
    def make_single_threshold_histograms(classifier_summary: ClassifierSummary, description=''):
        font_name = "Arial"
        font_size = 14

        sleep_threshold = 1 - Constants.WAKE_THRESHOLD
        for feature_set in classifier_summary.performance_dictionary:

            raw_performances = classifier_summary.performance_dictionary[feature_set]
            performances = PerformanceSummarizer.apply_single_threshold(raw_performances,
                                                                        sleep_threshold=sleep_threshold)
            number_of_subjects = len(performances)

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
            all_accuracies = np.zeros((number_of_subjects, 1))
            all_fraction_wake_correct = np.zeros((number_of_subjects, 1))
            all_fraction_sleep_correct = np.zeros((number_of_subjects, 1))
            all_kappas = np.zeros((number_of_subjects, 1))

            for subject_index in range(number_of_subjects):
                all_accuracies[subject_index, 0] = performances[subject_index].accuracy
                all_fraction_wake_correct[subject_index, 0] = performances[subject_index].wake_correct
                all_fraction_sleep_correct[subject_index, 0] = performances[subject_index].sleep_correct
                all_kappas[subject_index, 0] = performances[subject_index].kappa

            dt = 0.02
            ax[0, 0].hist(all_accuracies, bins=np.arange(0, 1 + dt, dt), color="skyblue", ec="skyblue")
            ax[0, 0].set_xlabel('Accuracy', fontsize=font_size, fontname=font_name)
            ax[0, 0].set_ylabel('Count', fontsize=font_size, fontname=font_name)
            ax[0, 0].set_xlim((0, 1))

            ax[0, 1].hist(all_kappas, bins=np.arange(0, 1 + dt, dt), color="skyblue", ec="skyblue")
            ax[0, 1].set_xlabel('Cohen\'s Kappa', fontsize=font_size, fontname=font_name)
            ax[0, 1].set_xlim((0, 1))

            ax[1, 0].hist(all_fraction_wake_correct, bins=np.arange(0, 1 + dt, dt), color="skyblue", ec="skyblue")
            ax[1, 0].set_xlabel('Fraction wake correct (specificity)', fontsize=font_size, fontname=font_name)
            ax[1, 0].set_ylabel('Count', fontsize=font_size, fontname=font_name)
            ax[1, 0].set_xlim((0, 1))

            ax[1, 1].hist(all_fraction_sleep_correct, bins=np.arange(0, 1 + dt, dt), color="skyblue", ec="skyblue")
            ax[1, 1].set_xlabel('Fraction sleep correct (sensitivity)', fontsize=font_size, fontname=font_name)
            ax[1, 1].set_xlim((0, 1))
            plt.tight_layout()
            file_save_name = str(
                Constants.FIGURE_FILE_PATH) + '/figure_' + classifier_summary.attributed_classifier.name + '_' + \
                             description + '_single_threshold_histograms.png'

            plt.savefig(file_save_name, dpi=300)
            plt.close()

    @staticmethod
    def make_bland_altman(classifier_summary: ClassifierSummary, description=''):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))

        wake_threshold = Constants.WAKE_THRESHOLD
        rem_threshold = Constants.REM_THRESHOLD
        
        for feature_set in classifier_summary.performance_dictionary:
            raw_performances = classifier_summary.performance_dictionary[feature_set]

            number_of_subjects = len(raw_performances)

            plot_color = FeatureSetService.get_color(feature_set)

            for subject_index in range(number_of_subjects):
                raw_performance = raw_performances[subject_index]
                true_labels = raw_performance.true_labels
                predicted_labels = PerformanceBuilder.apply_threshold_three_class(raw_performance, wake_threshold,
                                                                                  rem_threshold)

                actual_sol = SleepMetricsCalculator.get_sleep_onset_latency(true_labels)
                predicted_sol = SleepMetricsCalculator.get_sleep_onset_latency(predicted_labels)

                sol_diff = (actual_sol - predicted_sol)
                ax[0, 0].scatter(actual_sol, sol_diff, c=plot_color)

                ax[0, 0].set_xlabel("SOL")
                ax[0, 0].set_ylabel("Difference in SOL")

                actual_waso = SleepMetricsCalculator.get_wake_after_sleep_onset(true_labels)
                predicted_waso = SleepMetricsCalculator.get_wake_after_sleep_onset(predicted_labels)

                waso_diff = (actual_waso - predicted_waso)

                ax[0, 1].scatter(actual_waso, waso_diff, c=plot_color)
                ax[0, 1].set_xlabel("WASO")

                ax[0, 1].set_ylabel("Difference in WASO")

                actual_tst = SleepMetricsCalculator.get_tst(true_labels)
                predicted_tst = SleepMetricsCalculator.get_tst(predicted_labels)

                tst_diff = (actual_tst - predicted_tst)
                ax[1, 0].scatter(actual_tst, tst_diff, c=plot_color)

                ax[1, 0].set_xlabel("TST")
                ax[1, 0].set_ylabel("Difference in TST")

                actual_sleep_efficiency = SleepMetricsCalculator.get_sleep_efficiency(true_labels)
                predicted_sleep_efficiency = SleepMetricsCalculator.get_sleep_efficiency(predicted_labels)

                sleep_efficiency_diff = (actual_sleep_efficiency - predicted_sleep_efficiency)
                ax[1, 1].scatter(actual_sleep_efficiency, sleep_efficiency_diff, c=plot_color)
                ax[1, 1].set_xlabel("Sleep efficiency")

                ax[1, 1].set_ylabel("Difference in sleep efficiency")

                actual_time_in_rem = SleepMetricsCalculator.get_time_in_rem(true_labels)
                predicted_time_in_rem = SleepMetricsCalculator.get_time_in_rem(predicted_labels)

                time_in_rem_diff = (actual_time_in_rem - predicted_time_in_rem)
                ax[2, 0].scatter(actual_time_in_rem, time_in_rem_diff, c=plot_color)
                ax[2, 0].set_xlabel("Time in REM")
                ax[2, 0].set_ylabel("Difference in time in REM")

                actual_time_in_nrem = SleepMetricsCalculator.get_time_in_nrem(true_labels)
                predicted_time_in_nrem = SleepMetricsCalculator.get_time_in_nrem(predicted_labels)

                time_in_nrem_diff = (actual_time_in_nrem - predicted_time_in_nrem)

                ax[2, 1].set_xlabel("Time in NREM")
                ax[2, 1].set_ylabel("Difference in time in NREM")
                font = font_manager.FontProperties(family='Arial', style='normal', size=10)
                if subject_index == 0:
                    ax[2, 1].scatter(actual_time_in_nrem, time_in_nrem_diff, c=plot_color,
                                     label=FeatureSetService.get_label(feature_set))
                    ax[2, 1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, prop=font)
                else:
                    ax[2, 1].scatter(actual_time_in_nrem, time_in_nrem_diff, c=plot_color)

        plt.tight_layout()
        file_save_name = str(
            Constants.FIGURE_FILE_PATH) + '/figure_' + classifier_summary.attributed_classifier.name + '_' + \
                         description + '_bland_altman.png'

        plt.savefig(file_save_name, dpi=300)
        plt.close()
