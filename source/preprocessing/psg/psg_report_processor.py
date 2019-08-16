import docx2txt as docx2txt

from source import utils
from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.report_summary import ReportSummary


class PSGReportProcessor(object):

    @staticmethod
    def get_start_epoch_for_subject(subject_id):
        if int(subject_id) < 38:
            return 1
        if int(subject_id) == 40:
            return 35
        if int(subject_id) == 39:
            return 32
        if int(subject_id) == 38:
            return 37
        if int(subject_id) == 42:
            return 27
        if int(subject_id) == 41:
            return 21

    @staticmethod
    def get_summary_from_pdf(report_file_path):
        report_raw_text = utils.convert_pdf_to_txt(str(report_file_path), True)
        raw_text_split_at_epoch = report_raw_text.split('Epoch')

        split_at_study_date = (raw_text_split_at_epoch[0]).split('Study Date:  ')
        study_date = (split_at_study_date[1]).split(' \n')[0]

        split_at_zero = (raw_text_split_at_epoch[1]).split('\n\n0')

        split_at_newline = (split_at_zero[1]).split('\n')
        split_at_colon = (raw_text_split_at_epoch[1]).split(':')

        start_epoch = split_at_newline[1]
        start_time = split_at_colon[0][-2:] + ':' + split_at_colon[1] + ':' + split_at_colon[2][:5]

        return ReportSummary(study_date=study_date, start_time=start_time, start_epoch=start_epoch,
                             file_type=PSGFileType.Compumedics)

    @staticmethod
    def get_summary_from_docx(report_file_path):
        report_text = docx2txt.process(report_file_path)
        report_split = report_text.split('DATE: ')
        date = report_split[1].split('\n')[0]

        return ReportSummary(study_date=date, start_time=None, start_epoch=1, file_type=PSGFileType.Vitaport)
