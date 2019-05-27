import os.path

ANNOTATION_TOOL_PATH = os.path.dirname(os.path.realpath(__file__))
CALIB_DATA_PATH = os.path.join(ANNOTATION_TOOL_PATH, 'calib_data')
DATA_PATH = os.path.join(ANNOTATION_TOOL_PATH, 'data')

NUM_CLASSES = 4