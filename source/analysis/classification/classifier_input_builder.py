import numpy as np

from source.analysis.setup.sleep_labeler import SleepLabeler


class ClassifierInputBuilder(object):

    @staticmethod
    def get_array(subject_ids, subject_dictionary, feature_set):

        all_subjects_features = np.array([])
        all_subjects_labels = np.array([])

        for subject_id in subject_ids:
            subject_features = np.array([])
            subject = subject_dictionary[subject_id]
            feature_dictionary = subject.feature_dictionary

            for feature in feature_set:
                feature_data = feature_dictionary[feature]
                subject_features = ClassifierInputBuilder.__append_feature(subject_features, feature_data)

            all_subjects_features = ClassifierInputBuilder.__stack(all_subjects_features, subject_features)
            all_subjects_labels = ClassifierInputBuilder.__stack(all_subjects_labels, subject.labeled_sleep)

        return all_subjects_features, all_subjects_labels

    @staticmethod
    def get_sleep_wake_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_sleep_wake(raw_labels)
        return values, processed_labels

    @staticmethod
    def get_three_class_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_three_class(raw_labels)
        return values, processed_labels

    @staticmethod
    def __append_feature(array, feature):
        if len(np.shape(feature)) < 2:
            feature = np.transpose([feature])
        if np.shape(array)[0] == 0:
            array = feature
        else:
            array = np.hstack((array, feature))

        return array

    @staticmethod
    def __stack(combined_array, new_array):
        if np.shape(combined_array)[0] == 0:
            combined_array = new_array
        else:
            combined_array = np.vstack((combined_array, new_array))
        return combined_array
