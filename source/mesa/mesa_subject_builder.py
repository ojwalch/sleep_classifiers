import matplotlib.pyplot as plt
import numpy as np

from source import utils
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.mesa.mesa_psg_service import MesaPSGService
from source.mesa.mesa_time_based_service import MesaTimeBasedService
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.interval import Interval
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService


class MesaSubjectBuilder(object):

    @staticmethod
    def build(file_id):
        if Constants.VERBOSE:
            print('Building MESA subject ' + file_id + '...')

        raw_labeled_sleep = MesaPSGService.load_raw(file_id)
        heart_rate_collection = MesaHeartRateService.load_raw(file_id)
        activity_count_collection = MesaActigraphyService.load_raw(file_id)
        circadian_model = MesaTimeBasedService.load_circadian_model(file_id)

        if activity_count_collection.data[0][0] != -1 and circadian_model is not None:

            circadian_model = utils.remove_nans(circadian_model)

            interval = Interval(start_time=0, end_time=np.shape(raw_labeled_sleep)[0])

            activity_count_collection = ActivityCountService.crop(activity_count_collection, interval)
            heart_rate_collection = HeartRateService.crop(heart_rate_collection, interval)

            valid_epochs = []

            for timestamp in range(interval.start_time, interval.end_time, Epoch.DURATION):
                epoch = Epoch(timestamp=timestamp, index=len(valid_epochs))
                activity_count_indices = ActivityCountFeatureService.get_window(activity_count_collection.timestamps,
                                                                                epoch)
                heart_rate_indices = HeartRateFeatureService.get_window(heart_rate_collection.timestamps, epoch)

                if len(activity_count_indices) > 0 and 0 not in heart_rate_collection.values[heart_rate_indices]:
                    valid_epochs.append(epoch)
                else:
                    pass

            labeled_sleep = np.expand_dims(
                MesaPSGService.crop(psg_labels=raw_labeled_sleep, valid_epochs=valid_epochs),
                axis=1)

            feature_count = ActivityCountFeatureService.build_from_collection(activity_count_collection,
                                                                              valid_epochs)
            feature_hr = HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)
            feature_time = np.expand_dims(TimeBasedFeatureService.build_time(valid_epochs), axis=1)
            feature_cosine = np.expand_dims(TimeBasedFeatureService.build_cosine(valid_epochs), axis=1)

            feature_circadian = TimeBasedFeatureService.build_circadian_model_from_raw(circadian_model,
                                                                                       valid_epochs)
            feature_dictionary = {FeatureType.count: feature_count,
                                  FeatureType.heart_rate: feature_hr,
                                  FeatureType.time: feature_time,
                                  FeatureType.circadian_model: feature_circadian,
                                  FeatureType.cosine: feature_cosine}

            subject = Subject(subject_id=file_id,
                              labeled_sleep=labeled_sleep,
                              feature_dictionary=feature_dictionary)

            # Uncomment to save files for all subjects
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
            # plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(file_id + '.png')))
            # plt.close()

            return subject
