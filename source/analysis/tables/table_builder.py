from source.analysis.performance.performance_summarizer import PerformanceSummarizer
from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.feature_set_service import FeatureSetService
from source.analysis.setup.feature_type import FeatureType


class TableBuilder(object):
    pass

    @staticmethod
    def print_table_sw(classifier_summary):
        performance_dictionary = classifier_summary.performance_dictionary
        attributed_classifier = classifier_summary.attributed_classifier

        frontmatter = '\\begin{table}  \\caption{Sleep/wake differentiation performance by ' \
                      + attributed_classifier.name \
                      + ' across different feature inputs in the Apple Watch (PPG, MEMS) dataset} ' \
                        '\\begin{tabular}{l*{5}{c}} & Accuracy & Wake correct (specificity) ' \
                        '& Sleep correct (sensitivity) & $\\kappa$ & AUC \\\\ '

        print(frontmatter)
        for feature_set in performance_dictionary:
            raw_performances: [RawPerformance] = performance_dictionary[feature_set]

            print('\\hline ' + FeatureSetService.get_label(list(feature_set)) + ' &')

            true_positive_thresholds, performance_summaries = PerformanceSummarizer.summarize_thresholds(
                raw_performances)

            for true_positive_threshold, performance in zip(true_positive_thresholds, performance_summaries):
                if true_positive_threshold == 0.8:
                    print(
                        str(round(performance.accuracy, 3)) + ' & ' +
                        str(round(performance.wake_correct, 3)) + ' & ' +
                        str(round(performance.sleep_correct, 3)) + ' & ' +
                        str(round(performance.kappa, 3)) + ' & ' +
                        str(round(performance.auc, 3)) + '  \\\\')
                else:
                    print('& ' +
                          str(round(performance.accuracy, 3)) + ' & ' +
                          str(round(performance.wake_correct, 3)) + ' & ' +
                          str(round(performance.sleep_correct, 3))
                          + ' & ' + str(round(performance.kappa, 3)) + ' &   \\\\')

        backmatter = '\\hline \\end{tabular}  \\label{tab:' + \
                     attributed_classifier.name[0:4] + \
                     'params} \\small \\vspace{.2cm} ' \
                     '\\caption*{Fraction of wake correct, fraction of sleep correct, accuracy, ' \
                     '$\\kappa$, and AUC for sleep-wake predictions of ' + attributed_classifier.name + \
                     ' with use of motion, HR, clock proxy, or combination of features. PPG, ' \
                     'photoplethysmography; MEMS, microelectromechanical systems; HR, heart rate; ' \
                     'AUC, area under the curve.} \\end{table}'

        print(backmatter)

    @staticmethod
    def print_table_three_class(classifier_summaries):

        frontmatter = '\\begin{tabular}{l | l | c | c | c | c | c } & & Wake Correct & NREM' \
                      ' Correct & REM Correct & Best accuracy & $\\kappa$ \\'

        print(frontmatter)

        for classifier_summary in classifier_summaries:
            performance_dictionary = classifier_summary.performance_dictionary
            attributed_classifier = classifier_summary.attributed_classifier

            for feature_set in performance_dictionary:

                if list(feature_set) == [FeatureType.count]:
                    print('\\hline ' + attributed_classifier.name + ' & ' + FeatureSetService.get_label(
                        list(feature_set)) + ' & ')

                else:
                    print(' & ' + FeatureSetService.get_label(list(feature_set)) + ' & ')

                three_class_performance_summary = performance_dictionary[feature_set]
                wake_correct = three_class_performance_summary.wake_correct
                nrem_correct = three_class_performance_summary.nrem_correct
                rem_correct = three_class_performance_summary.rem_correct
                accuracy = three_class_performance_summary.accuracy
                kappa = three_class_performance_summary.kappa

                print(str(round(wake_correct, 3)) + ' & ' +
                      str(round(nrem_correct, 3)) + ' & ' +
                      str(round(rem_correct, 3)) + ' & ' +
                      str(round(accuracy, 3)) + ' & ' +
                      str(round(kappa, 3)) + '\\\\')

        backmatter = '\\end{tabular}  \\label{tab:REM_params} \\small \\vspace{.2cm}'

        print(backmatter)
