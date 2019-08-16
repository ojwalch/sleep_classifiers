from unittest import TestCase, mock
from unittest.mock import MagicMock
import numpy as np

from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.psg_label_service import PSGLabelService
from source.preprocessing.psg.psg_service import PSGService


class TestPSGLabelService(TestCase):

    @mock.patch('source.preprocessing.psg.psg_label_service.pd')
    def test_load(self, mock_pd):
        mock_pd.read_csv.return_value = mock_return = MagicMock()
        mock_return.values = expected_return = np.array([1, 2, 3, 4, 5])
        actual_returned_value = PSGLabelService.load("subjectA")

        self.assertListEqual(expected_return.tolist(), actual_returned_value.tolist())
        mock_pd.read_csv.assert_called_once_with(str(PSGLabelService.get_path("subjectA")))

    def test_get_path(self):
        expected_path = Constants.FEATURE_FILE_PATH.joinpath("subjectA" + '_psg_labels.out')

        self.assertEqual(expected_path, PSGLabelService.get_path("subjectA"))

    @mock.patch.object(PSGService, 'load_cropped_array')
    def test_build(self, mock_load_cropped_array):
        subject_id = 'subjectA'
        data = np.array(
            [[1, 1], [10, 2], [20, 0], [40, 1], [70, 2], [90, 3], [100, 1], [120, 2]])

        valid_epochs = [Epoch(timestamp=10, index=1), Epoch(timestamp=40, index=2)]
        mock_load_cropped_array.return_value = data
        expected_labels = np.array([2, 1])

        returned_labels = PSGLabelService.build(subject_id, valid_epochs)

        self.assertEqual(expected_labels.tolist(), returned_labels.tolist())

    @mock.patch.object(PSGLabelService, 'get_path')
    @mock.patch('source.preprocessing.psg.psg_label_service.np')
    def test_write(self, mock_np, mock_get_path):
        labels = np.array([1, 2, 3])
        mock_get_path.return_value = path = 'path/to/return'
        PSGLabelService.write("subjectA", labels)

        mock_np.savetxt.assert_called_once_with(path, labels, fmt='%f')
