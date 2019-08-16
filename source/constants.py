from source import utils


class Constants(object):
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = False
    CROPPED_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    FEATURE_FILE_PATH = utils.get_project_root().joinpath('outputs/features/')
    FIGURE_FILE_PATH = utils.get_project_root().joinpath('outputs/figures/')
    LOWER_BOUND = -0.2
