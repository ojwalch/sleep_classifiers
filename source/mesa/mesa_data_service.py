from source.mesa.mesa_subject_builder import MesaSubjectBuilder
from source.mesa.metadata_service import MetadataService


class MesaDataService(object):

    @staticmethod
    def get_all_subjects():
        all_files = MetadataService.get_all_files()
        all_subjects = []
        for file in all_files:
            file_id = file[-8:-4]
            subject = MesaSubjectBuilder.build(file_id)
            if subject is not None:
                all_subjects.append(subject)

        return all_subjects
