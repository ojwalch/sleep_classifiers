from unittest import TestCase

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.feature_set_service import FeatureSetService
import seaborn as sns


class TestFeatureSetService(TestCase):

    def test_get_feature_set_labels(self):
        self.assertEqual("Motion only", FeatureSetService.get_label([FeatureType.count]))
        self.assertEqual("HR only", FeatureSetService.get_label([FeatureType.heart_rate]))
        self.assertEqual("Motion, HR", FeatureSetService.get_label([FeatureType.count,
                                                                    FeatureType.heart_rate]))
        self.assertEqual("Motion, HR, and Clock", FeatureSetService.get_label([FeatureType.count,
                                                                               FeatureType.heart_rate,
                                                                               FeatureType.circadian_model]))
        self.assertEqual("Motion, HR, and Time", FeatureSetService.get_label([FeatureType.count,
                                                                              FeatureType.heart_rate,
                                                                              FeatureType.time
                                                                              ]))
        self.assertEqual("Motion, HR, and Cosine", FeatureSetService.get_label([FeatureType.count,
                                                                                FeatureType.heart_rate,
                                                                                FeatureType.cosine
                                                                                ]))

    def test_get_feature_set_colors(self):
        self.assertEqual(sns.xkcd_rgb["denim blue"], FeatureSetService.get_color([FeatureType.count]))
        self.assertEqual(sns.xkcd_rgb["yellow orange"], FeatureSetService.get_color([FeatureType.heart_rate]))
        self.assertEqual(sns.xkcd_rgb["medium green"], FeatureSetService.get_color([FeatureType.count,
                                                                                    FeatureType.heart_rate]))
        self.assertEqual(sns.xkcd_rgb["medium pink"], FeatureSetService.get_color([FeatureType.count,
                                                                                   FeatureType.heart_rate,
                                                                                   FeatureType.circadian_model]))
        self.assertEqual(sns.xkcd_rgb["greyish"], FeatureSetService.get_color([FeatureType.count,
                                                                               FeatureType.heart_rate,
                                                                               FeatureType.time
                                                                               ]))
        self.assertEqual(sns.xkcd_rgb["plum"], FeatureSetService.get_color([FeatureType.count,
                                                                            FeatureType.heart_rate,
                                                                            FeatureType.cosine
                                                                            ]))
