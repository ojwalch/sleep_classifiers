from source.analysis.classification.classifier_service import ClassifierService
from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.data_split import DataSplit
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter
from source.mesa.mesa_data_service import MesaDataService


class SleepWakeClassifierSummaryBuilder(object):

    @staticmethod
    def build_monte_carlo(attributed_classifier: AttributedClassifier, feature_sets: [[FeatureType]],
                          number_of_splits: int) -> ClassifierSummary:
        subject_ids = SubjectBuilder.get_all_subject_ids()
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.by_fraction(subject_ids, test_fraction=0.3, number_of_splits=number_of_splits)

        return SleepWakeClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                  attributed_classifier,
                                                                  feature_sets)

    @staticmethod
    def build_leave_one_out(attributed_classifier: AttributedClassifier,
                            feature_sets: [[FeatureType]]) -> ClassifierSummary:
        subject_ids = SubjectBuilder.get_all_subject_ids()
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.leave_one_out(subject_ids)

        return SleepWakeClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                  attributed_classifier,
                                                                  feature_sets)

    @staticmethod
    def run_feature_sets(data_splits: [DataSplit], subject_dictionary, attributed_classifier: AttributedClassifier,
                         feature_sets: [[FeatureType]]):
        performance_dictionary = {}
        for feature_set in feature_sets:
            raw_performance_results = ClassifierService.run_sw(data_splits, attributed_classifier,
                                                               subject_dictionary, feature_set)
            performance_dictionary[tuple(feature_set)] = raw_performance_results

        return ClassifierSummary(attributed_classifier, performance_dictionary)

    @staticmethod
    def build_mesa(attributed_classifier: AttributedClassifier, feature_sets: [[FeatureType]]):
        apple_watch_subjects = SubjectBuilder.get_subject_dictionary()
        mesa_subjects = MesaDataService.get_all_subjects()
        training_set = []
        testing_set = []
        mesa_dictionary = {}

        for subject_key in apple_watch_subjects:
            training_set.append(subject_key)

        for mesa_subject in mesa_subjects:
            mesa_subject.subject_id = 'mesa' + mesa_subject.subject_id
            testing_set.append(mesa_subject.subject_id)
            mesa_dictionary[mesa_subject.subject_id] = mesa_subject

        data_split = DataSplit(training_set=training_set, testing_set=testing_set)
        apple_watch_subjects.update(mesa_dictionary)

        return SleepWakeClassifierSummaryBuilder.run_feature_sets([data_split], apple_watch_subjects,
                                                                  attributed_classifier,
                                                                  feature_sets)


class ThreeClassClassifierSummaryBuilder(object):

    @staticmethod
    def build_monte_carlo(attributed_classifier: AttributedClassifier, feature_sets: [[FeatureType]],
                          number_of_splits: int) -> ClassifierSummary:
        subject_ids = SubjectBuilder.get_all_subject_ids()
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.by_fraction(subject_ids, test_fraction=0.3, number_of_splits=number_of_splits)

        return ThreeClassClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                   attributed_classifier,
                                                                   feature_sets)

    @staticmethod
    def build_leave_one_out(attributed_classifier: AttributedClassifier,
                            feature_sets: [[FeatureType]]) -> ClassifierSummary:
        subject_ids = SubjectBuilder.get_all_subject_ids()
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.leave_one_out(subject_ids)

        return ThreeClassClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                   attributed_classifier,
                                                                   feature_sets)

    @staticmethod
    def run_feature_sets(data_splits: [DataSplit], subject_dictionary, attributed_classifier: AttributedClassifier,
                         feature_sets: [[FeatureType]], use_preloaded=False):
        performance_dictionary = {}
        for feature_set in feature_sets:
            if use_preloaded:
                raw_performance_results = ClassifierService.run_three_class_with_loaded_model(data_splits,
                                                                                              attributed_classifier,
                                                                                              subject_dictionary,
                                                                                              feature_set)
            else:
                raw_performance_results = ClassifierService.run_three_class(data_splits, attributed_classifier,
                                                                            subject_dictionary, feature_set)
            performance_dictionary[tuple(feature_set)] = raw_performance_results

        return ClassifierSummary(attributed_classifier, performance_dictionary)

    @staticmethod
    def build_mesa_leave_one_out(attributed_classifier: AttributedClassifier, feature_sets: [[FeatureType]]):
        apple_watch_subjects = SubjectBuilder.get_subject_dictionary()
        mesa_subjects = MesaDataService.get_all_subjects()
        training_set = []
        mesa_dictionary = {}
        data_splits = []

        for subject_key in apple_watch_subjects:
            training_set.append(subject_key)

        for mesa_subject in mesa_subjects:
            mesa_subject.subject_id = 'mesa' + mesa_subject.subject_id
            mesa_dictionary[mesa_subject.subject_id] = mesa_subject
            testing_set = [mesa_subject.subject_id]
            data_split = DataSplit(training_set=training_set, testing_set=testing_set)
            data_splits.append(data_split)

        apple_watch_subjects.update(mesa_dictionary)

        return ThreeClassClassifierSummaryBuilder.run_feature_sets(data_splits, apple_watch_subjects,
                                                                   attributed_classifier,
                                                                   feature_sets, True)

    @staticmethod
    def build_mesa_all_combined(attributed_classifier: AttributedClassifier, feature_sets: [[FeatureType]]):
        apple_watch_subjects = SubjectBuilder.get_subject_dictionary()
        mesa_subjects = MesaDataService.get_all_subjects()
        training_set = []
        testing_set = []
        mesa_dictionary = {}

        for subject_key in apple_watch_subjects:
            training_set.append(subject_key)

        for mesa_subject in mesa_subjects:
            mesa_subject.subject_id = 'mesa' + mesa_subject.subject_id
            mesa_dictionary[mesa_subject.subject_id] = mesa_subject
            testing_set.append(mesa_subject.subject_id)

        data_split = DataSplit(training_set=training_set, testing_set=testing_set)
        apple_watch_subjects.update(mesa_dictionary)

        return ThreeClassClassifierSummaryBuilder.run_feature_sets([data_split], apple_watch_subjects,
                                                                   attributed_classifier,
                                                                   feature_sets)
