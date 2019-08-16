from source.analysis.performance.epoch_performance import SleepWakePerformance, ThreeClassPerformance
from source.analysis.performance.performance_builder import PerformanceBuilder
from source.analysis.performance.raw_performance import RawPerformance


class PerformanceSummarizer(object):

    @staticmethod
    def average(sleep_wake_performance_array: [SleepWakePerformance]):
        average_performance = SleepWakePerformance(
            accuracy=0,
            wake_correct=0,
            sleep_correct=0,
            auc=0,
            kappa=0,
            wake_predictive_value=0,
            sleep_predictive_value=0
        )

        for sleep_wake_performance in sleep_wake_performance_array:
            average_performance.accuracy += sleep_wake_performance.accuracy
            average_performance.wake_correct += sleep_wake_performance.wake_correct
            average_performance.sleep_correct += sleep_wake_performance.sleep_correct
            average_performance.auc += sleep_wake_performance.auc
            average_performance.kappa += sleep_wake_performance.kappa
            average_performance.wake_predictive_value += sleep_wake_performance.wake_predictive_value
            average_performance.sleep_predictive_value += sleep_wake_performance.sleep_predictive_value

        number_of_performance_models = len(sleep_wake_performance_array)

        average_performance.accuracy = PerformanceSummarizer.__calculate_average(average_performance.accuracy,
                                                                                 number_of_performance_models)

        average_performance.wake_correct = PerformanceSummarizer.__calculate_average(average_performance.wake_correct,
                                                                                     number_of_performance_models)

        average_performance.sleep_correct = PerformanceSummarizer.__calculate_average(
            average_performance.sleep_correct,
            number_of_performance_models)

        average_performance.auc = PerformanceSummarizer.__calculate_average(average_performance.auc,
                                                                            number_of_performance_models)

        average_performance.kappa = PerformanceSummarizer.__calculate_average(average_performance.kappa,
                                                                              number_of_performance_models)

        average_performance.wake_predictive_value = PerformanceSummarizer.__calculate_average(
            average_performance.wake_predictive_value,
            number_of_performance_models)

        average_performance.sleep_predictive_value = PerformanceSummarizer.__calculate_average(
            average_performance.sleep_predictive_value,
            number_of_performance_models)

        return average_performance

    @staticmethod
    def average_three_class(three_class_performance_array: [ThreeClassPerformance]):
        average_performance = ThreeClassPerformance(accuracy=0,
                                                    wake_correct=0,
                                                    rem_correct=0,
                                                    nrem_correct=0,
                                                    kappa=0)

        for three_class_performance in three_class_performance_array:
            average_performance.accuracy += three_class_performance.accuracy
            average_performance.wake_correct += three_class_performance.wake_correct
            average_performance.rem_correct += three_class_performance.rem_correct
            average_performance.nrem_correct += three_class_performance.nrem_correct
            average_performance.kappa += three_class_performance.kappa

        number_of_performance_models = len(three_class_performance_array)

        average_performance.accuracy = PerformanceSummarizer.__calculate_average(average_performance.accuracy,
                                                                                 number_of_performance_models)

        average_performance.wake_correct = PerformanceSummarizer.__calculate_average(average_performance.wake_correct,
                                                                                     number_of_performance_models)

        average_performance.rem_correct = PerformanceSummarizer.__calculate_average(
            average_performance.rem_correct,
            number_of_performance_models)

        average_performance.nrem_correct = PerformanceSummarizer.__calculate_average(average_performance.nrem_correct,
                                                                                     number_of_performance_models)

        average_performance.kappa = PerformanceSummarizer.__calculate_average(average_performance.kappa,
                                                                              number_of_performance_models)

        return average_performance

    @staticmethod
    def summarize_thresholds(raw_performances: [RawPerformance]):
        performance_summaries = []

        true_positive_thresholds = [0.8, 0.9, 0.93, 0.95]
        for fixed_true_positive_rate in true_positive_thresholds:

            performances = []
            for raw_performance in raw_performances:
                performance = PerformanceBuilder.build_with_true_positive_rate_threshold(raw_performance,
                                                                                         fixed_true_positive_rate)
                performances.append(performance)

            performance_summary = PerformanceSummarizer.average(performances)
            performance_summaries.append(performance_summary)

        return true_positive_thresholds, performance_summaries

    @staticmethod
    def apply_single_threshold(raw_performances: [RawPerformance], sleep_threshold):
        performances = []

        for raw_performance in raw_performances:
            performance = PerformanceBuilder.build_with_sleep_threshold(raw_performance, sleep_threshold)
            performances.append(performance)

        return performances

    @staticmethod
    def __calculate_average(value, count):
        return value / count
