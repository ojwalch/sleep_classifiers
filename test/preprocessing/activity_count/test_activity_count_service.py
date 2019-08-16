from unittest import TestCase, mock

from source import utils
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_collection import ActivityCountCollection
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
import numpy as np

from test.test_helper import TestHelper


class TestActivityCountService(TestCase):

    @mock.patch.object(ActivityCountService, 'load')
    @mock.patch.object(ActivityCountService, 'get_cropped_file_path')
    def test_load_cropped(self, mock_cropped_file_path, mock_load):
        subject_id = 'subjectA'
        mock_cropped_file_path.return_value = path = 'path/to/file'
        data = np.array([[1, 2, 3], [4, 5, 6]])
        expected_activity_count_collection = ActivityCountCollection(subject_id=subject_id, data=data)
        mock_load.return_value = data
        returned_activity_count_collection = ActivityCountService.load_cropped(subject_id)
        mock_cropped_file_path.assert_called_once_with(subject_id)
        mock_load.assert_called_once_with(path)
        TestHelper.assert_models_equal(self, expected_activity_count_collection, returned_activity_count_collection)

    @mock.patch('source.preprocessing.activity_count.activity_count_service.pd')
    def test_load(self, mock_pd):
        counts_file_path = 'path/to/file'
        mock_pd.read_csv.return_value.values = expected_data = np.array([1, 2, 3, 4, 5])

        returned_data = ActivityCountService.load(counts_file_path)

        mock_pd.read_csv.assert_called_once_with(counts_file_path)
        self.assertEqual(expected_data.tolist(), returned_data.tolist())

    def test_get_cropped_file_path(self):
        subject_id = 'subject1'
        file_path = ActivityCountService.get_cropped_file_path(subject_id)
        self.assertEqual(Constants.CROPPED_FILE_PATH.joinpath("subject1_cleaned_counts.out"), file_path)

    @mock.patch('source.preprocessing.activity_count.activity_count_service.os')
    def test_build_activity_counts(self, mock_os):
        expected_argument = 'matlab -nodisplay -nosplash -nodesktop -r \"run(\'' + str(
            utils.get_project_root()) + '/source/make_counts.m\'); exit;\"'

        ActivityCountService.build_activity_counts()

        mock_os.system.assert_called_once_with(expected_argument)

