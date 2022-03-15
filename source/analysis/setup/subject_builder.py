from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service \
    import \
    ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import \
    HeartRateFeatureService
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.preprocessing.time.time_based_feature_service import \
    TimeBasedFeatureService


class SubjectBuilder(object):

    @staticmethod
    def get_all_subject_ids():

        subjects_as_ints = [3509524, 5132496, 1066528, 5498603, 2638030,
                            2598705, 5383425, 1455390, 4018081, 9961348,
                            1449548, 8258170, 781756, 9106476, 8686948,
                            8530312, 3997827, 4314139, 1818471, 4426783,
                            8173033, 7749105, 5797046, 759667, 8000685,
                            6220552, 844359, 9618981, 1360686, 46343,
                            8692923]

        subjects_as_strings = []

        for subject in subjects_as_ints:
            subjects_as_strings.append(str(subject))
        return subjects_as_strings

    @staticmethod
    def get_all_disordered_subject_ids():
        # This is excluding anyone without a 100% full night, and all the nones
        return ["d02", "d03", "d04", "d05", "d08", "d09",
                "d10", "d11", "d12", "d13", "d15", "d16", "d18",
                "d19", "d21", "d23", "d24", "d25", "d28",
                "d29", "d30", "d32", "d34", "d35", "d36", "d37",
                "d38", "d39", "d40"]

        # return ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09",
        #         "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18",
        #         "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d28",
        #         "d29", "d30", "d31", "d32", "d33", "d34", "d35", "d36", "d37",
        #         "d38", "d39", "d40"]

    @staticmethod
    def get_apnea_only_sleepers():
        base_ids = SubjectBuilder.get_all_disordered_subject_ids()
        apnea_only_ids = []
        for id in base_ids:
            diagnoses = SubjectBuilder.subject_to_disorder(id)
            if 'modosa' in diagnoses or 'milosa' in diagnoses or 'sevosa' in\
                diagnoses:
                apnea_only_ids.append(id)

        return apnea_only_ids

    @staticmethod
    def get_subject_dictionary():
        subject_dictionary = {}
        all_subject_ids = SubjectBuilder.get_all_subject_ids()
        for subject_id in all_subject_ids:
            subject_dictionary[subject_id] = SubjectBuilder.build(subject_id)

        return subject_dictionary

    @staticmethod
    def get_subject_dictionary_disordered():
        subject_dictionary = {}
        all_subject_ids = SubjectBuilder.get_all_disordered_subject_ids()
        for subject_id in all_subject_ids:
            subject_dictionary[subject_id] = SubjectBuilder.build(subject_id)

        return subject_dictionary

    @staticmethod
    def build(subject_id):
        feature_count = ActivityCountFeatureService.load(subject_id)
        feature_hr = HeartRateFeatureService.load(subject_id)
        feature_time = TimeBasedFeatureService.load_time(subject_id)
        if Constants.INCLUDE_CIRCADIAN:
            feature_circadian = TimeBasedFeatureService.load_circadian_model(
                subject_id)
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
        # plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(subject_id +
        # '_applewatch.png')))
        # plt.close()
        return subject

    @staticmethod
    def subject_to_disorder(subject_id):

        subject_disorder_dictionary = {
            'd01' : [],
            'd02' : ['narcolepsy', 'modosa'],
            'd03': ['narcolepsy', 'modosa'],
            'd04': ['modosa'],
            'd05': ['narcolepsy'],
            'd06': ['sevosa'],
            'd07': ['sevosa'],
            'd08': ['mildosa'],
            'd09': ['modosa'],
            'd10': ['mildosa'],
            'd11': ['mildosa'],
            'd12': ['mildosa'],
            'd13': ['modosa'],
            'd14': ['modosa'],
            'd15': ['mildosa'],
            'd16': ['mildosa'],
            'd17': ['mildosa'],
            'd18': ['mildosa'],
            'd19': ['modosa'],
            'd20': ['mildosa'],
            'd21': ['modosa'],
            'd22': ['mildosa'],
            'd23': ['mildosa'],
            'd24': ['modosa'],
            'd25': ['modosa'],
            'd26': [],
            'd28': ['mildosa'],
            'd29': ['mildosa'],
            'd30': ['mildosa'],
            'd31': [],
            'd32': ['mildosa'],
            'd33': [],
            'd34': ['mildosa'],
            'd35': ['mildosa'],
            'd36': ['mildosa'],
            'd37': ['modosa'],
            'd38': ['sevosa'],
            'd39': ['modosa'],
            'd40': ['modosa']
        }
        return subject_disorder_dictionary[subject_id]