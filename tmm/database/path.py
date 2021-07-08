import os


def path():
    '''
    Gets the current directory path and returns it.
    '''
    folder_path = os.path.dirname(os.path.realpath(__file__))

    return folder_path + os.sep
