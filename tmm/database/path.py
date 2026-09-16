import os


def path():
    """Return the database folder path with a trailing separator."""
    folder_path = os.path.dirname(os.path.realpath(__file__))

    return folder_path + os.sep
