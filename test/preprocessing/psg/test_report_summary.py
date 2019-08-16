from unittest import TestCase

from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary


class TestReportSummary(TestCase):
    def test_constructor(self):
        start_time = "12:00:02 AM"
        start_epoch = 4
        study_date = "11/2/2019"
        file_type = PSGFileType.Vitaport
        report_summary = ReportSummary(study_date=study_date, start_epoch=start_epoch, start_time=start_time,
                                       file_type=file_type)

        self.assertEqual(report_summary.study_date, study_date)
        self.assertEqual(report_summary.start_epoch, start_epoch)
        self.assertEqual(report_summary.start_time, start_time)
        self.assertEqual(report_summary.file_type, file_type)
