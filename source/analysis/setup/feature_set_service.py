import seaborn as sns

from source.analysis.setup.feature_type import FeatureType


class FeatureSetService(object):

    @staticmethod
    def get_label(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.count}:
            return 'Motion only'
        if set(feature_set) == {FeatureType.heart_rate}:
            return 'HR only'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate}:
            return 'Motion, HR'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model}:
            return 'Motion, HR, and Clock'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return 'Motion, HR, and Cosine'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.time}:
            return 'Motion, HR, and Time'

    @staticmethod
    def get_color(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.count}:
            return sns.xkcd_rgb["denim blue"]
        if set(feature_set) == {FeatureType.heart_rate}:
            return sns.xkcd_rgb["yellow orange"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate}:
            return sns.xkcd_rgb["medium green"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model}:
            return sns.xkcd_rgb["medium pink"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return sns.xkcd_rgb["plum"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.time}:
            return sns.xkcd_rgb["greyish"]
