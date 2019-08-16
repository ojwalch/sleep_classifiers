from unittest import TestCase, mock

from mock import MagicMock

from source.mesa.mesa_psg_service import MesaPSGService


class TestMesaPSGService(TestCase):

    @mock.patch('source.mesa.mesa_psg_service.minidom')
    @mock.patch('source.mesa.mesa_psg_service.np')
    @mock.patch('source.mesa.mesa_psg_service.utils')
    def test_load_raw(self, mock_utils, mock_np, mock_minidom):
        file_id = "2"
        mock_utils.get_project_root.return_value = "project/root"
        mock_minidom.parse.return_value = mock_xml_document = MagicMock()
        first_scored_event = MagicMock()
        second_scored_event = MagicMock()
        third_scored_event = MagicMock()

        child_nodes = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                       MagicMock()]
        child_nodes[3].childNodes[0].nodeValue = 'Stage 4 sleep|4'
        child_nodes[5].childNodes[0].nodeValue = 0.0
        child_nodes[7].childNodes[0].nodeValue = 90.0

        first_scored_event.childNodes = child_nodes

        child_nodes = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                       MagicMock()]
        child_nodes[3].childNodes[0].nodeValue = 'REM sleep|5'
        child_nodes[5].childNodes[0].nodeValue = 90.0
        child_nodes[7].childNodes[0].nodeValue = 30.0

        second_scored_event.childNodes = child_nodes

        child_nodes = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                       MagicMock()]
        child_nodes[3].childNodes[0].nodeValue = 'Stage 4 sleep|4'
        child_nodes[5].childNodes[0].nodeValue = 120.0
        child_nodes[7].childNodes[0].nodeValue = 900.0

        third_scored_event.childNodes = child_nodes

        scored_events = [first_scored_event,
                         second_scored_event,
                         third_scored_event]

        mock_xml_document.getElementsByTagName.return_value = scored_events

        expected_stages = []
        for i in range(90):
            expected_stages.append(4)
        for i in range(30):
            expected_stages.append(5)
        for i in range(900):
            expected_stages.append(4)
        mock_np.array.return_value = expected_return = [1, 10, 100]

        stages = MesaPSGService.load_raw(file_id)

        mock_utils.get_project_root.assert_called_once()

        mock_np.array.assert_called_once_with(expected_stages)
        self.assertListEqual(expected_return, stages)
