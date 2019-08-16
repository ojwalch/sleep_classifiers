from unittest import TestCase, mock
from unittest.mock import call

from source.mesa.mesa_data_service import MesaDataService


class TestMesaDataService(TestCase):

    @mock.patch('source.mesa.mesa_data_service.MesaSubjectBuilder')
    @mock.patch('source.mesa.mesa_data_service.MetadataService')
    def test_get_all_subjects(self, mock_metadata_service, mock_subject_builder):
        mock_metadata_service.get_all_files.return_value = ["file1-3243.edf", "file2-1234.edf"]

        mock_subject_builder.build.side_effect = expected_subjects = ["A", "B"]

        subjects = MesaDataService.get_all_subjects()

        mock_subject_builder.build.assert_has_calls([call("3243"), call("1234")])
        self.assertListEqual(expected_subjects, subjects)
