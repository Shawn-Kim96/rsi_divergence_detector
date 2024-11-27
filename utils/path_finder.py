import os

PROJECT_NAME = 'rsi_divergence_detector'

def get_project_path():
    curpath = os.path.abspath('.')
    return curpath.split(PROJECT_NAME)[0] + PROJECT_NAME

