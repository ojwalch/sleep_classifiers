from unittest import TestCase, mock
import numpy as np

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.analysis.setup.subject_builder import SubjectBuilder
from test.test_helper import TestHelper


class TestSubjectBuilder(TestCase):
    def test_get_all_subject_ids(self):
        self.assertListEqual(
            ['1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '19', '20', '22', '23', '25',
             '27', '28', '29', '30', '32', '33', '34', '35', '38', '39', '41', '42'],
            SubjectBuilder.get_all_subject_ids())

    @mock.patch('source.analysis.setup.subject_builder.PSGLabelService')
    @mock.patch('source.analysis.setup.subject_builder.TimeBasedFeatureService')
    @mock.patch('source.analysis.setup.subject_builder.HeartRateFeatureService')
    @mock.patch('source.analysis.setup.subject_builder.ActivityCountFeatureService')
    def test_build(self, mock_activity_count_fs, mock_heart_rate_fs, mock_time_based_fs, mock_psg_label_service):
        subject_id = "subjectA"
        mock_activity_count_fs.get_path_to_file.return_value = activity_count_feature = np.array([1, 2, 3, 4, 5])
        mock_heart_rate_fs.get_path_to_file.return_value = heart_rate_feature = np.array([6, 7, 8, 9, 10])
        mock_time_based_fs.load_time.return_value = time_feature = np.array([11])
        mock_time_based_fs.load_circadian_model.return_value = circadian_feature = np.array([12])
        mock_time_based_fs.load_cosine.return_value = cosine = np.array([12])
        mock_psg_label_service.load.return_value = labels = np.array([13])

        feature_dictionary = {FeatureType.count: activity_count_feature,
                              FeatureType.heart_rate: heart_rate_feature,
                              FeatureType.time: time_feature,
                              FeatureType.circadian_model: circadian_feature,
                              FeatureType.cosine: cosine
                              }

        expected_subject = Subject(subject_id=subject_id, labeled_sleep=labels, feature_dictionary=feature_dictionary)
        returned_subject = SubjectBuilder.build(subject_id)

        TestHelper.assert_models_equal(self, expected_subject, returned_subject)

    @mock.patch.object(SubjectBuilder, 'build')
    @mock.patch.object(SubjectBuilder, 'get_all_subject_ids')
    def test_build_subject_dictionary(self, mock_subject_ids, mock_build):
        mock_subject_ids.return_value = subject_ids = ["subject123"]
        mock_build.return_value = placeholder = np.array([1, 2, 3])

        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        mock_build.assert_called_once_with(subject_ids[0])
        self.assertDictEqual({subject_ids[0]: placeholder}, subject_dictionary)
