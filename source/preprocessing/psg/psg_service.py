import csv

import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.psg.compumedics_processor import CompumedicsProcessor
from source.preprocessing.psg.psg_converter import PSGConverter
from source.preprocessing.psg.psg_file_type import PSGFileType
from source.preprocessing.psg.psg_raw_data_collection import PSGRawDataCollection
from source.preprocessing.psg.psg_report_processor import PSGReportProcessor
from source.preprocessing.psg.stage_item import StageItem
from source.preprocessing.psg.vitaport_processor import VitaportProcessor


class PSGService(object):

    @staticmethod
    def get_path_to_file(subject_id):
        psg_dir = utils.get_project_root().joinpath('data/psg')
        compumedics_file = psg_dir.joinpath('compumedics/AW0' + subject_id.zfill(2) + '.TXT')
        if compumedics_file.is_file():
            return compumedics_file

        txt_file = psg_dir.joinpath('vitaport/AW0' + subject_id.zfill(2) + '011.txt')
        if txt_file.is_file():
            return txt_file

    @staticmethod
    def get_type_and_report(subject_id):
        report_dir = utils.get_project_root().joinpath('data/reports')

        pdf_file = report_dir.joinpath('AW0' + subject_id.zfill(2) + '011_REPORT.pdf')
        if pdf_file.is_file():
            return pdf_file, PSGFileType.Compumedics

        docx_file = report_dir.joinpath('AW00' + subject_id + '011 Study Sleep Log.docx')
        if docx_file.is_file():
            return docx_file, PSGFileType.Vitaport

    @staticmethod
    def read_raw(subject_id):
        psg_stage_path = PSGService.get_path_to_file(subject_id)
        psg_report_path, psg_type = PSGService.get_type_and_report(subject_id)

        if psg_type == PSGFileType.Compumedics:
            report_summary = PSGReportProcessor.get_summary_from_pdf(psg_report_path)
            report_summary.start_epoch = PSGReportProcessor.get_start_epoch_for_subject(subject_id)

            data = CompumedicsProcessor.parse(report_summary, psg_stage_path)
            return PSGRawDataCollection(subject_id=subject_id, data=data)

        if psg_type == PSGFileType.Vitaport:
            report_summary = PSGReportProcessor.get_summary_from_docx(psg_report_path)
            data = VitaportProcessor.parse(report_summary, psg_stage_path)
            return PSGRawDataCollection(subject_id=subject_id, data=data)

    @staticmethod
    def read_precleaned(subject_id):
        psg_path = str(utils.get_project_root().joinpath('data/labels/' + subject_id + '_labeled_sleep.txt'))
        data = []

        with open(psg_path, 'rt') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            count = 0
            rows_per_epoch = 1
            for row in file_reader:
                if count == 0:
                    start_time = float(row[0])
                    start_score = int(row[1])
                    epoch = Epoch(timestamp=start_time, index=1)
                    data.append(StageItem(epoch=epoch, stage=PSGConverter.get_label_from_int(start_score)))
                else:
                    timestamp = start_time + count * 30
                    score = int(row[1])
                    epoch = Epoch(timestamp=timestamp,
                                  index=(1 + int(np.floor(count / rows_per_epoch))))

                    data.append(StageItem(epoch=epoch, stage=PSGConverter.get_label_from_int(score)))
                count = count + 1
        return PSGRawDataCollection(subject_id=subject_id, data=data)

    @staticmethod
    def crop(psg_raw_collection, interval):
        subject_id = psg_raw_collection.subject_id

        stage_items = []
        for stage_item in psg_raw_collection.data:
            timestamp = stage_item.epoch.timestamp
            if interval.start_time <= timestamp < interval.end_time:
                stage_items.append(stage_item)

        return PSGRawDataCollection(subject_id=subject_id, data=stage_items)

    @staticmethod
    def write(psg_raw_data_collection):
        data_array = []

        for index in range(len(psg_raw_data_collection.data)):
            stage_item = psg_raw_data_collection.data[index]
            data_array.append([stage_item.epoch.timestamp, stage_item.stage.value])

        np_psg_array = np.array(data_array)
        psg_output_path = Constants.CROPPED_FILE_PATH.joinpath(psg_raw_data_collection.subject_id + "_cleaned_psg.out")

        np.savetxt(psg_output_path, np_psg_array, fmt='%f')

    @staticmethod
    def load_cropped_array(subject_id):
        cropped_psg_path = Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_psg.out")
        return pd.read_csv(str(cropped_psg_path), delimiter=' ').values

    @staticmethod
    def load_cropped(subject_id):
        cropped_array = PSGService.load_cropped_array(subject_id)
        stage_items = []

        for row in range(np.shape(cropped_array)[0]):
            value = cropped_array[row, 1]
            stage_items.append(StageItem(epoch=Epoch(timestamp=cropped_array[row, 0], index=row),
                                         stage=PSGConverter.get_label_from_int(value)))

        return PSGRawDataCollection(subject_id=subject_id, data=stage_items)
