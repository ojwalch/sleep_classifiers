import math

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import font_manager

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.performance.curve_performance_builder import CurvePerformanceBuilder
from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.setup.feature_set_service import FeatureSetService
from source.constants import Constants


class CurvePlotBuilder(object):

    @staticmethod
    def tidy_plot():
        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    @staticmethod
    def build_roc_plot(classifier_summary: ClassifierSummary, positive_class=1):
        for feature_set in classifier_summary.performance_dictionary:
            raw_performances = classifier_summary.performance_dictionary[feature_set]
            roc_performance = CurvePerformanceBuilder.build_roc_from_raw(raw_performances, positive_class)

            legend_text = FeatureSetService.get_label(list(feature_set))
            plot_color = FeatureSetService.get_color(list(feature_set))

            plt.plot(roc_performance.false_positive_rates, roc_performance.true_positive_rates,
                     label=legend_text, color=plot_color)

    @staticmethod
    def make_roc_sw(classifier_summary: ClassifierSummary, description=''):
        CurvePlotBuilder.build_roc_plot(classifier_summary)
        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of wake scored as sleep',
                                    'Fraction of sleep scored as sleep', (1.0, 0.4))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_sw_roc.png')))
        plt.close()

    @staticmethod
    def make_roc_one_vs_rest(classifier_summary: ClassifierSummary, description=''):
        CurvePlotBuilder.build_roc_plot(classifier_summary, 0)
        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of REM or NREM scored as wake',
                                    'Fraction of wake scored as wake', (1.0, 0.4))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_ovr_wake_roc.png')))
        plt.close()

        CurvePlotBuilder.build_roc_plot(classifier_summary, 1)
        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of wake or REM scored as NREM',
                                    'Fraction of NREM scored as NREM', (1.0, 0.4))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_ovr_nrem_roc.png')))
        plt.close()

        CurvePlotBuilder.build_roc_plot(classifier_summary, 2)
        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of wake or NREM scored as REM',
                                    'Fraction of REM scored as REM', (1.0, 0.4))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_ovr_rem_roc.png')))
        plt.close()

    @staticmethod
    def build_pr_plot(classifier_summary: ClassifierSummary):
        for feature_set in classifier_summary.performance_dictionary:
            raw_performances = classifier_summary.performance_dictionary[feature_set]
            roc_performance = CurvePerformanceBuilder.build_precision_recall_from_raw(raw_performances)

            legend_text = FeatureSetService.get_label(list(feature_set))
            plot_color = FeatureSetService.get_color(list(feature_set))

            plt.plot(roc_performance.recalls, roc_performance.precisions,
                     label=legend_text, color=plot_color)

    @staticmethod
    def make_pr_sw(classifier_summary: ClassifierSummary, description=''):
        CurvePlotBuilder.build_pr_plot(classifier_summary)
        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of wake scored as wake',
                                    'Fraction of predicted wake correct', (0.5, 1.0))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_sw_pr.png')))
        plt.close()

    @staticmethod
    def make_three_class_roc(classifier_summary: ClassifierSummary, description=''):

        performance_dictionary = {}

        for feature_set in classifier_summary.performance_dictionary:
            raw_performances = classifier_summary.performance_dictionary[feature_set]
            sleep_wake_roc_performance, rem_roc_performance, nrem_roc_performance, three_class_performances = \
                CurvePerformanceBuilder.build_three_class_roc_with_binary_search(
                    raw_performances)

            performance_dictionary[feature_set] = PerformanceSummarizer.average_three_class(three_class_performances)

            legend_text = FeatureSetService.get_label(list(feature_set))
            plot_color = FeatureSetService.get_color(list(feature_set))

            plt.plot(sleep_wake_roc_performance.false_positive_rates, sleep_wake_roc_performance.true_positive_rates,
                     label=legend_text, color=plot_color)
            plt.plot(nrem_roc_performance.false_positive_rates, nrem_roc_performance.true_positive_rates,
                     color=plot_color, linestyle=':')
            plt.plot(rem_roc_performance.false_positive_rates, rem_roc_performance.true_positive_rates,
                     color=plot_color, linestyle='--')

        CurvePlotBuilder.tidy_plot()
        CurvePlotBuilder.set_labels(classifier_summary.attributed_classifier,
                                    'Fraction of wake scored as REM or NREM',
                                    'Fraction of REM, NREM scored correctly', (1.0, 0.4))

        number_of_trials = len(next(iter(classifier_summary.performance_dictionary.values())))
        plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(
            classifier_summary.attributed_classifier.name + '_' + str(
                number_of_trials) + '_' + description + '_three_class_roc.png')))
        plt.close()

        return performance_dictionary

    @staticmethod
    def set_labels(attributed_classifier, x_label_text, y_label_text, legend_location):
        font_name = "Arial"
        font_size = 14
        font = font_manager.FontProperties(family=font_name, style='normal', size=font_size)

        if attributed_classifier.name == 'Neural Net':
            plt.legend(bbox_to_anchor=legend_location, borderaxespad=0., prop=font)

        plt.xlabel(x_label_text, fontsize=font_size, fontname=font_name)
        plt.ylabel(y_label_text, fontsize=font_size, fontname=font_name)

        plt.title(attributed_classifier.name, fontsize=18, fontname=font_name, fontweight='bold')

    @staticmethod
    def combine_plots_as_grid(classifiers, number_of_trials, plot_extension):
        combined_filenames = []
        for attributed_classifier in classifiers:
            combined_filenames.append(str(Constants.FIGURE_FILE_PATH) + '/' +
                                      attributed_classifier.name + '_' +
                                      str(number_of_trials) + '_' + plot_extension + '.png')

        images = list(map(Image.open, combined_filenames))
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        max_height = max(heights)

        new_image = Image.new('RGB', (2 * max_width, 2 * max_height))

        count = 0
        for im in images:
            x_offset = int((count % 2) * max_width)
            y_offset = int(math.floor(count / 2) * max_height)

            new_image.paste(im, (x_offset, y_offset))
            count = count + 1

        new_image.save(str(Constants.FIGURE_FILE_PATH) + '/figure_' + str(number_of_trials) + plot_extension + '.png')

    @staticmethod
    def combine_sw_and_three_class_plots(attributed_classifier, number_of_trials, plot_extension):
        combined_filenames = [str(Constants.FIGURE_FILE_PATH) + '/' +
                              attributed_classifier.name + '_' +
                              str(number_of_trials) + '__' + plot_extension + '_sw_roc.png',
                              str(Constants.FIGURE_FILE_PATH) + '/' +
                              attributed_classifier.name + '_' +
                              str(number_of_trials) + '__' + plot_extension + '_three_class_roc.png']

        images = list(map(Image.open, combined_filenames))
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        max_height = max(heights)

        new_image = Image.new('RGB', (2 * max_width, 1 * max_height))

        count = 0
        for im in images:
            x_offset = int((count % 2) * max_width)
            y_offset = int(math.floor(count / 2) * max_height)

            new_image.paste(im, (x_offset, y_offset))
            count = count + 1

        new_image.save(
            str(Constants.FIGURE_FILE_PATH) + '/figure_' + attributed_classifier.name + str(
                number_of_trials) + plot_extension + '_combined.png')
