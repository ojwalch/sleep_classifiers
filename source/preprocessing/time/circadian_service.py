import os

from source import utils


class CircadianService(object):

    @staticmethod
    def build_circadian_model():
        os.system('matlab -nodisplay -nosplash -nodesktop -r \"run(\'' + str(
            utils.get_project_root()) + '/source/preprocessing/time/clock_proxy/runCircadianModel.m\'); exit;\"')

    @staticmethod
    def build_circadian_mesa():
        os.system('matlab -nodisplay -nosplash -nodesktop -r \"run(\'' + str(
            utils.get_project_root()) + '/source/preprocessing/time/clock_proxy/runCircadianMESA.m\'); exit;\"')
