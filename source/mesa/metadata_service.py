import csv
import glob

import numpy as np

from source import utils


class MetadataService(object):

    @staticmethod
    def get_all_files():
        project_root = str(utils.get_project_root())
        return glob.glob(project_root + "/data/mesa/polysomnography/edfs/*edf")

    @staticmethod
    def get_metadata_dictionary():
        metadata_dictionary = {}
        ref_dict = dict()
        ref_dict['ahi'] = 'ahiu35'
        ref_dict['age'] = 'sleepage5c'
        ref_dict['gender'] = 'gender1'  # 0 Female, 1 Male
        ref_dict['tst'] = 'slpprdp5'
        ref_dict['tib'] = 'time_bed5'
        ref_dict['waso'] = 'waso5'
        ref_dict['slp_eff'] = 'slp_eff5'
        ref_dict['time_rem'] = 'timerem5'
        ref_dict['time_n1'] = 'timest15'
        ref_dict['time_n2'] = 'timest25'
        ref_dict['time_n34'] = 'timest345'

        with open('../mesa/mesa-sleep-dataset-0.3.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            is_first_row = True
            for row in csv_reader:
                if is_first_row:
                    is_first_row = False
                    count = 0
                    for col in row:
                        if col in ref_dict.values():
                            keys = list(ref_dict.keys())
                            key = keys[list(ref_dict.values()).index(col)]
                            ref_dict[key] = count  # Replace with index number where data is
                        count = count + 1
                else:
                    subject_dict = {}
                    for key in ref_dict:
                        val = row[ref_dict[key]]
                        if len(val) > 0:
                            subject_dict[key] = float(val)
                        else:
                            subject_dict[key] = np.nan
                    metadata_dictionary[int(row[0])] = subject_dict

        return metadata_dictionary

    @staticmethod
    def print_table(subject_ids):
        all_files = MetadataService.get_all_files()
        metadata_dictionary = MetadataService.get_metadata_dictionary()
        ahis = []
        ages = []
        genders = []
        tst = []
        tib = []
        waso = []
        slp_eff = []
        time_rem = []
        time_nrem = []

        for subject_index in subject_ids:  # Get all metadata for subjects
            subject = int(all_files[subject_index][-8:-4])
            subject_dict = metadata_dictionary[subject]
            ahis.append(subject_dict['ahi'])
            ages.append(subject_dict['age'])
            genders.append(subject_dict['gender'])
            tst.append(subject_dict['tst'])
            tib.append(subject_dict['tib'])
            waso.append(subject_dict['waso'])
            slp_eff.append(subject_dict['slp_eff'])
            time_rem.append(subject_dict['time_rem'])
            time_nrem.append(
                float(subject_dict['time_n1']) + float(subject_dict['time_n2']) + float(subject_dict['time_n34']))

        ahis = np.array(ahis)
        ages = np.array(ages)
        genders = np.array(genders)
        tst = np.array(tst)
        tib = np.array(tib)
        waso = np.array(waso)
        slp_eff = np.array(slp_eff)

        print('N women: ' + str(len(genders) - np.count_nonzero(genders)))

        latex = True
        if latex:
            print(
                '\\begin{table} \\caption{Sleep summary statistics - MESA subcohort}  \\label{tab:mesa} \\small  \\begin{tabularx}{\\columnwidth}{X | X | X  }\\hline Parameter & Mean (SD) & Range \\\\ \\hline')
        else:
            print('Parameter, Mean (SD), Range')

        print(MetadataService.data_to_line('Age (years)', ages, latex))
        print(MetadataService.data_to_line('TST (minutes)', tst, latex))
        print(MetadataService.data_to_line('TIB (minutes)', tib, latex))
        print(MetadataService.data_to_line('WASO (minutes)', waso, latex))
        print(MetadataService.data_to_line('SE (\%)', slp_eff, latex))
        print(MetadataService.data_to_line('AHI', ahis, latex))

        if latex:
            print('\\end{tabularx} \\end{table}')

    @staticmethod
    def data_to_line(name, data, latex=False):
        if latex:
            return name + ' & ' + str(round(np.mean(data), 2)) + ' (' + str(round(np.std(data), 2)) + ') & ' \
                   + str(round(np.min(data), 2)) + '-' + str(round(np.max(data), 2)) + '\\\\'
        else:
            return name + ',' + str(round(np.mean(data), 2)) + ' (' + str(round(np.std(data), 2)) + '),' \
                   + str(round(np.min(data), 2)) + '-' + str(round(np.max(data), 2))
