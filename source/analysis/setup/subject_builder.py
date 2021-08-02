from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService


class SubjectBuilder(object):

    @staticmethod
    def get_all_subject_ids():

        subjects_as_ints = [3509524, 5132496, 1066528, 5498603, 2638030, 2598705, 5383425, 1455390, 4018081, 9961348,
                            1449548, 8258170, 781756, 9106476, 8686948, 8530312, 3997827, 4314139, 1818471, 4426783,
                            8173033, 7749105, 5797046, 759667, 8000685, 6220552, 844359, 9618981, 1360686, 46343,
                            8692923]

        subjects_as_strings = []

        for subject in subjects_as_ints:
            subjects_as_strings.append(str(subject))
        return subjects_as_strings

    @staticmethod
    def get_subject_dictionary():
        subject_dictionary = {}
        all_subject_ids = SubjectBuilder.get_all_subject_ids()
        for subject_id in all_subject_ids:
            subject_dictionary[subject_id] = SubjectBuilder.build(subject_id)

        return subject_dictionary

    @staticmethod
    def build(subject_id):
        feature_count = ActivityCountFeatureService.load(subject_id)
        feature_hr = HeartRateFeatureService.load(subject_id)
        feature_time = TimeBasedFeatureService.load_time(subject_id)
        if Constants.INCLUDE_CIRCADIAN:
            feature_circadian = TimeBasedFeatureService.load_circadian_model(subject_id)
        else:
            feature_circadian = None
        feature_cosine = TimeBasedFeatureService.load_cosine(subject_id)
        labeled_sleep = PSGLabelService.load(subject_id)

        feature_dictionary = {FeatureType.count: feature_count,
                              FeatureType.heart_rate: feature_hr,
                              FeatureType.time: feature_time,
                              FeatureType.circadian_model: feature_circadian,
                              FeatureType.cosine: feature_cosine}

        subject = Subject(subject_id=subject_id,
                          labeled_sleep=labeled_sleep,
                          feature_dictionary=feature_dictionary)

        # Uncomment to save plots of every subject's data:
        # ax = plt.subplot(5, 1, 1)
        # ax.plot(range(len(feature_hr)), feature_hr)
        # ax = plt.subplot(5, 1, 2)
        # ax.plot(range(len(feature_count)), feature_count)
        # ax = plt.subplot(5, 1, 3)
        # ax.plot(range(len(feature_cosine)), feature_cosine)
        # ax = plt.subplot(5, 1, 4)
        # ax.plot(range(len(feature_circadian)), feature_circadian)
        # ax = plt.subplot(5, 1, 5)
        # ax.plot(range(len(labeled_sleep)), labeled_sleep)
        #
        # plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(subject_id + '_applewatch.png')))
        # plt.close()
        return subject
