from unittest import TestCase
from unittest.mock import call, MagicMock

import mock
import numpy as np
from sklearn.linear_model import LogisticRegression

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.figures.performance_plot_builder import PerformancePlotBuilder
from source.analysis.performance.curve_performance import ROCPerformance, PrecisionRecallPerformance
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from source.constants import Constants
from source.analysis.figures.curve_plot_builder import CurvePlotBuilder


class TestPlotBuilder(TestCase):
    @mock.patch('source.analysis.figures.performance_plot_builder.ImageFont')
    @mock.patch('source.analysis.figures.performance_plot_builder.ImageDraw')
    @mock.patch('source.analysis.figures.performance_plot_builder.Image')
    @mock.patch('source.analysis.figures.performance_plot_builder.PerformanceSummarizer')
    @mock.patch('source.analysis.figures.performance_plot_builder.np')
    @mock.patch('source.analysis.figures.performance_plot_builder.plt')
    def test_make_histogram_with_thresholds(self, mock_plt, mock_np, mock_performance_summarizer, mock_image,
                                            mock_image_draw, mock_image_font):
        raw_performances = ['raw_performance_placeholder_1', 'raw_performance_placeholder_2']

        performance_dictionary = {tuple([FeatureType.count]): raw_performances}
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        mock_ax = [[MagicMock(), MagicMock()],
                   [MagicMock(), MagicMock()],
                   [MagicMock(), MagicMock()],
                   [MagicMock(), MagicMock()]]

        number_of_subjects = 2
        mock_plt.subplots.return_value = 'placeholder1', mock_ax
        mock_np.zeros.side_effect = [np.zeros((number_of_subjects, 4)),
                                     np.zeros((number_of_subjects, 4))]

        dt = 0.02
        expected_range = np.arange(0, 1 + dt, dt)
        mock_np.arange.return_value = expected_range

        mock_subject1_threshold1 = MagicMock()
        mock_subject1_threshold2 = MagicMock()
        mock_subject1_threshold3 = MagicMock()
        mock_subject1_threshold4 = MagicMock()
        mock_subject2_threshold1 = MagicMock()
        mock_subject2_threshold2 = MagicMock()
        mock_subject2_threshold3 = MagicMock()
        mock_subject2_threshold4 = MagicMock()

        mock_subject1_threshold1.accuracy = 0
        mock_subject1_threshold1.wake_correct = 1
        mock_subject1_threshold2.accuracy = 2
        mock_subject1_threshold2.wake_correct = 3
        mock_subject1_threshold3.accuracy = 4
        mock_subject1_threshold3.wake_correct = 5
        mock_subject1_threshold4.accuracy = 6
        mock_subject1_threshold4.wake_correct = 7

        mock_subject2_threshold1.accuracy = 10
        mock_subject2_threshold1.wake_correct = 11
        mock_subject2_threshold2.accuracy = 12
        mock_subject2_threshold2.wake_correct = 13
        mock_subject2_threshold3.accuracy = 14
        mock_subject2_threshold3.wake_correct = 15
        mock_subject2_threshold4.accuracy = 16
        mock_subject2_threshold4.wake_correct = 17

        mock_performance_summarizer.summarize_thresholds.side_effect = [([1, 2, 3, 4], [mock_subject1_threshold1,
                                                                                        mock_subject1_threshold2,
                                                                                        mock_subject1_threshold3,
                                                                                        mock_subject1_threshold4]),
                                                                        ([1, 2, 3, 4], [mock_subject2_threshold1,
                                                                                        mock_subject2_threshold2,
                                                                                        mock_subject2_threshold3,
                                                                                        mock_subject2_threshold4])]

        file_save_name = str(
            Constants.FIGURE_FILE_PATH) + '/' + 'Motion only_Logistic Regression_histograms_with_thresholds.png'
        mock_image.open.return_value = mock_opened_image = MagicMock()
        mock_opened_image.size = 100, 200

        mock_image.new.return_value = new_image = MagicMock()
        mock_image_draw.Draw.return_value = image_draw = MagicMock()
        mock_image_font.truetype.return_value = font = "true type font"
        PerformancePlotBuilder.make_histogram_with_thresholds(classifier_summary)

        mock_plt.subplots.assert_called_once_with(
            nrows=4, ncols=2, figsize=(8, 8),
            sharex=True, sharey=True)

        mock_np.zeros.assert_has_calls(
            [call((2, 4)), call((2, 4))])
        mock_performance_summarizer.summarize_thresholds.assert_has_calls(
            [call([raw_performances[0]]),
             call([raw_performances[1]])])

        mock_ax[0][0].hist.assert_called_once_with([0, 10], bins=expected_range,
                                                   color="skyblue",
                                                   ec="skyblue")

        mock_ax[0][1].hist.assert_called_once_with([1, 11], bins=expected_range,
                                                   color="lightsalmon",
                                                   ec="lightsalmon")

        mock_ax[1][0].hist.assert_called_once_with([2, 12], bins=expected_range,
                                                   color="skyblue",
                                                   ec="skyblue")

        mock_ax[1][1].hist.assert_called_once_with([3, 13], bins=expected_range,
                                                   color="lightsalmon",
                                                   ec="lightsalmon")

        mock_ax[2][0].hist.assert_called_once_with([4, 14], bins=expected_range,
                                                   color="skyblue",
                                                   ec="skyblue")

        mock_ax[2][1].hist.assert_called_once_with([5, 15], bins=expected_range,
                                                   color="lightsalmon",
                                                   ec="lightsalmon")

        mock_ax[3][0].hist.assert_called_once_with([6, 16], bins=expected_range,
                                                   color="skyblue",
                                                   ec="skyblue")

        mock_ax[3][1].hist.assert_called_once_with([7, 17], bins=expected_range,
                                                   color="lightsalmon",
                                                   ec="lightsalmon")

        mock_ax[3][0].set_xlabel.assert_called_once_with('Accuracy', fontsize=16, fontname='Arial')
        mock_ax[3][1].set_xlabel.assert_called_once_with('Wake correct', fontsize=16, fontname='Arial')
        mock_ax[3][0].set_ylabel.assert_called_once_with('Count', fontsize=16, fontname='Arial')

        mock_ax[3][0].set_xlim.assert_called_once_with((0, 1))
        mock_ax[3][1].set_xlim.assert_called_once_with((0, 1))

        mock_plt.tight_layout.assert_called_once_with()
        mock_plt.savefig.assert_called_once_with(file_save_name,
                                                 dpi=300)
        mock_plt.close.assert_called_once_with()

        mock_image.open.assert_called_once_with(file_save_name)
        mock_image.new.assert_called_once_with('RGB', (int((1 + 0.3) * 100), 200), "white")

        new_image.paste.assert_called_once_with(mock_opened_image, (int(0.3 * 100), 0))
        mock_image_draw.Draw.assert_called_once_with(new_image)
        mock_image_font.truetype.assert_called_once_with('/Library/Fonts/Arial Unicode.ttf', 75)
        image_draw.text.assert_has_calls(
            [call((int(0.3 * 100 / 3), int((200 * 0.9) * 0.125)), "TPR = 0.8", (0, 0, 0), font=font),
             call((int(0.3 * 100 / 3), int((200 * 0.9) * 0.375)), "TPR = 0.9", (0, 0, 0), font=font),
             call((int(0.3 * 100 / 3), int((200 * 0.9) * 0.625)), "TPR = 0.93", (0, 0, 0), font=font),
             call((int(0.3 * 100 / 3), int((200 * 0.9) * 0.875)), "TPR = 0.95", (0, 0, 0), font=font)])
        new_image.save.assert_called_once_with(str(Constants.FIGURE_FILE_PATH) + '/' + 'figure_threshold_histogram.png')

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    def test_tidy_plot(self, mock_plt):
        mock_axis = mock_plt.subplot.return_value
        CurvePlotBuilder.tidy_plot()
        mock_plt.subplot.assert_called_once_with(111)

        mock_axis.spines[''].set_visible.assert_has_calls([call(False), call(False), call(True), call(True)])
        mock_axis.yaxis.set_ticks_position.assert_called_once_with('left')
        mock_axis.xaxis.set_ticks_position.assert_called_once_with('bottom')

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    @mock.patch('source.analysis.figures.curve_plot_builder.FeatureSetService')
    @mock.patch('source.analysis.figures.curve_plot_builder.CurvePerformanceBuilder')
    def test_plot_roc(self, mock_curve_performance_builder, mock_feature_set_service, mock_plt):
        first_raw_performances = [
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        second_raw_performances = [
            RawPerformance(true_labels=np.array([1, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 0]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        performance_dictionary = {tuple([FeatureType.count, FeatureType.heart_rate]): first_raw_performances,
                                  tuple([FeatureType.count]): second_raw_performances}

        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        first_roc_performance = ROCPerformance(false_positive_rates=np.array([1]), true_positive_rates=np.array([.2]))
        second_roc_performance = ROCPerformance(false_positive_rates=np.array([.3]), true_positive_rates=np.array([1]))

        first_label = 'Label 1'
        second_label = 'Label 2'
        first_color = '#ffffff'
        second_color = '#123456'

        mock_curve_performance_builder.build_roc_from_raw.side_effect = [first_roc_performance, second_roc_performance]
        mock_feature_set_service.get_label.side_effect = [first_label, second_label]
        mock_feature_set_service.get_color.side_effect = [first_color, second_color]

        CurvePlotBuilder.build_roc_plot(classifier_summary)

        mock_curve_performance_builder.build_roc_from_raw.assert_has_calls([call(first_raw_performances, 1)])
        mock_curve_performance_builder.build_roc_from_raw.assert_has_calls([call(second_raw_performances, 1)])

        mock_feature_set_service.get_label.assert_has_calls(
            [call([FeatureType.count, FeatureType.heart_rate])])

        mock_feature_set_service.get_color.assert_has_calls(
            [call([FeatureType.count, FeatureType.heart_rate])])

        mock_feature_set_service.get_label.assert_has_calls(
            [call([FeatureType.count])])

        mock_feature_set_service.get_color.assert_has_calls(
            [call([FeatureType.count])])

        mock_plt.plot.assert_has_calls(
            [call(first_roc_performance.false_positive_rates, first_roc_performance.true_positive_rates,
                  label=first_label, color=first_color)])
        mock_plt.plot.assert_has_calls(
            [call(second_roc_performance.false_positive_rates, second_roc_performance.true_positive_rates,
                  label=second_label, color=second_color)])

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    @mock.patch('source.analysis.figures.curve_plot_builder.FeatureSetService')
    @mock.patch('source.analysis.figures.curve_plot_builder.CurvePerformanceBuilder')
    def test_plot_pr(self, mock_curve_performance_builder, mock_feature_set_service, mock_plt):
        first_raw_performances = [
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 1]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        second_raw_performances = [
            RawPerformance(true_labels=np.array([1, 1]),
                           class_probabilities=np.array([[0, 1], [1, 0]])),
            RawPerformance(true_labels=np.array([0, 0]),
                           class_probabilities=np.array([[0.2, 0.8], [0.1, 0.9]]))]

        performance_dictionary = {tuple([FeatureType.count, FeatureType.heart_rate]): first_raw_performances,
                                  tuple([FeatureType.count]): second_raw_performances}

        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        first_pr_performance = PrecisionRecallPerformance(recalls=np.array([1]), precisions=np.array([.2]))
        second_pr_performance = PrecisionRecallPerformance(recalls=np.array([.3]), precisions=np.array([1]))

        first_label = 'Label 1'
        second_label = 'Label 2'
        first_color = '#ffffff'
        second_color = '#123456'

        mock_curve_performance_builder.build_precision_recall_from_raw.side_effect = [first_pr_performance,
                                                                                      second_pr_performance]
        mock_feature_set_service.get_label.side_effect = [first_label, second_label]
        mock_feature_set_service.get_color.side_effect = [first_color, second_color]

        CurvePlotBuilder.build_pr_plot(classifier_summary)

        mock_curve_performance_builder.build_precision_recall_from_raw.assert_has_calls([call(first_raw_performances)])
        mock_curve_performance_builder.build_precision_recall_from_raw.assert_has_calls([call(second_raw_performances)])

        mock_feature_set_service.get_label.assert_has_calls(
            [call([FeatureType.count, FeatureType.heart_rate])])

        mock_feature_set_service.get_color.assert_has_calls(
            [call([FeatureType.count, FeatureType.heart_rate])])

        mock_feature_set_service.get_label.assert_has_calls(
            [call([FeatureType.count])])

        mock_feature_set_service.get_color.assert_has_calls(
            [call([FeatureType.count])])

        mock_plt.plot.assert_has_calls(
            [call(first_pr_performance.recalls, first_pr_performance.precisions,
                  label=first_label, color=first_color)])
        mock_plt.plot.assert_has_calls(
            [call(second_pr_performance.recalls, second_pr_performance.precisions,
                  label=second_label, color=second_color)])

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    @mock.patch.object(CurvePlotBuilder, 'set_labels')
    @mock.patch.object(CurvePlotBuilder, 'tidy_plot')
    @mock.patch.object(CurvePlotBuilder, 'build_roc_plot')
    def test_make_roc_plot(self, mock_build_roc, mock_tidy_plot, mock_set_labels, mock_plt):
        performance_dictionary = {
            tuple([FeatureType.count, FeatureType.heart_rate]): ['placeholder', 'for', 'raw', 'performances'],
            tuple([FeatureType.count]): ['placeholder', 'for', 'raw', 'performances']}

        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        CurvePlotBuilder.make_roc_sw(classifier_summary)

        mock_build_roc.assert_called_once_with(classifier_summary)
        mock_tidy_plot.assert_called_once_with()
        mock_set_labels.assert_called_once_with(attributed_classifier,
                                                'Fraction of wake scored as sleep',
                                                'Fraction of sleep scored as sleep',
                                                (1.0, 0.4))

        mock_plt.savefig.assert_called_once_with(str(
            Constants.FIGURE_FILE_PATH.joinpath(
                attributed_classifier.name + '_' + str(4) + '__sw_roc.png')))
        mock_plt.close.assert_called_once_with()

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    @mock.patch.object(CurvePlotBuilder, 'set_labels')
    @mock.patch.object(CurvePlotBuilder, 'tidy_plot')
    @mock.patch.object(CurvePlotBuilder, 'build_pr_plot')
    def test_make_pr_plot(self, mock_build_pr, mock_tidy_plot, mock_set_labels, mock_plt):
        performance_dictionary = {
            tuple([FeatureType.count, FeatureType.heart_rate]): ['placeholder', 'for', 'raw', 'performances'],
            tuple([FeatureType.count]): ['placeholder', 'for', 'raw', 'performances']}

        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        classifier_summary = ClassifierSummary(attributed_classifier=attributed_classifier,
                                               performance_dictionary=performance_dictionary)

        CurvePlotBuilder.make_pr_sw(classifier_summary)

        mock_build_pr.assert_called_once_with(classifier_summary)
        mock_tidy_plot.assert_called_once_with()
        mock_set_labels.assert_called_once_with(attributed_classifier,
                                                'Fraction of wake scored as wake',
                                                'Fraction of predicted wake correct', (0.5, 1.0))

        mock_plt.savefig.assert_called_once_with(str(
            Constants.FIGURE_FILE_PATH.joinpath(
                attributed_classifier.name + '_' + str(4) + '__sw_pr.png')))
        mock_plt.close.assert_called_once_with()

    @mock.patch('source.analysis.figures.curve_plot_builder.plt')
    @mock.patch('source.analysis.figures.curve_plot_builder.font_manager')
    def test_set_labels(self, mock_font_manager, mock_plt):
        attributed_classifier = AttributedClassifier(name="Logistic Regression", classifier=LogisticRegression())
        x_label = 'X Label Text'
        y_label = 'Y Label Text'
        legend_location = (1.0, 0.2)
        mock_font_manager.FontProperties.return_value = font_placeholder = 'FontPlaceholder'

        CurvePlotBuilder.set_labels(attributed_classifier, x_label, y_label, legend_location)

        font_name = "Arial"
        font_size = 14
        mock_font_manager.FontProperties.assert_called_once_with(family=font_name, style='normal', size=font_size)

        mock_plt.xlabel.assert_called_once_with(x_label, fontsize=font_size, fontname=font_name)
        mock_plt.ylabel.assert_called_once_with(y_label, fontsize=font_size, fontname=font_name)
        mock_plt.title.assert_called_once_with(attributed_classifier.name, fontsize=18, fontname=font_name,
                                               fontweight='bold')

        attributed_classifier = AttributedClassifier(name="Neural Net", classifier=LogisticRegression())

        CurvePlotBuilder.set_labels(attributed_classifier, x_label, y_label, legend_location)

        mock_plt.legend.assert_called_once_with(bbox_to_anchor=legend_location, borderaxespad=0., prop=font_placeholder)
