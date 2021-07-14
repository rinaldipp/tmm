import numpy as np
import time
import h5py


def save_dict_to_hdf5(dic, key, h5file):
    """
    Saves dictionary into HDF5 file.

    Parameters
    ----------
    dic : dictionary
        Python dictionary that will be saved into the h5 file.
    key : string
        Dictionary key.
    h5file : h5py.File
        Output .h5 file that is already open.

    Returns
    -------
    Nothing.
    """
    recursively_save_dict_contents_to_group(h5file, key + '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively saves dictionary into HDF5 group. Called when a dictionary has other dictionaries inside it.

    Parameters
    ----------
    h5file : h5py.File
        Output .h5 file that is already open.
    path : string
        h5 group path.
    dic : dictionary
        Python dictionary that will be saved into the h5 file.

    Returns
    -------
    Nothing.
    """
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + str(key) + '/', item)
        elif item is None:
            h5file[path + key] = 'None'
        else:
            if '/' in key:
                key = key.replace('/', '_div_')
            h5file[path + key] = item


def load_dict_from_hdf5(h5file, key):
    """
    Load dictionary from HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Input .h5 file that is already open.
    key : string
        Dictionary key.

    Returns
    -------
    recursively_load_dict_contents_from_group : dictionary
        Dictionary that can contain other dictionaries inside.
    """
    return recursively_load_dict_contents_from_group(h5file, key + '/')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively loads dictionaries from HDF5 group.

    Parameters
    ----------
    h5file : h5py.File
        Input .h5 file that is already open.
    path : string
        h5 group path.

    Returns
    -------
    ans : dictionary
        Dictionary containing the values inside the h5 group.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            item = parse_dataset_item(item)
            if '_div_' in key:
                key = key = key.replace('_div_', '/')
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            dict_key = eval(key) if len(key) == 1 else key
            ans[dict_key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def parse_dataset_item(item):
    """
    Parses a HDF5 Dataset based on the datatype.

    Parameters
    ----------
    item : h5py.Dataset
        Dataset containing data that will be parsed.

    Returns
    -------
    item : array, string, None, int, float or bool
        Parsed dataset.
    """
    item = np.array(item, dtype=item.dtype)
    if item.dtype == 'object':
        item = item.tolist().decode('UTF-8')
        if item == 'None':
            item = None
    elif np.issubdtype(item.dtype, np.integer):
        item = int(item)
    elif np.issubdtype(item.dtype, np.float) and item.shape == ():
        item = float(item)
    elif np.issubdtype(item.dtype, np.bool):
        item = bool(item)
    return item


def save_class_to_hdf5(self, filename='class', ext='.h5', folder=None, timestamp=False):
    """
    Saves a Class into a HDF5 file.

    Parameters
    ----------
    self : Class
        Python Class object.
    filename : string, optional
        Output filename.
    ext : string, optional
        Output extension.
    folder : None or string, optional
        Output folder. If 'None' is passed the current folder is used.
    timestamp : bool, optional
        Boolean to apply timestamping to the output filename.

    Returns
    -------
    Nothing.
    """
    timestr = time.strftime("%Y%m%d-%H%M_")

    if folder is not None:
        if timestamp is True:
            outfile = folder + timestr + filename + ext
        else:
            outfile = folder + filename + ext
    else:
        if timestamp is True:
            outfile = timestr + filename + ext
        else:
            outfile = filename + ext

    with h5py.File(outfile, 'w') as hdf:
        for attr, value in vars(self).items():
            if isinstance(value, dict) is False and value is not None:
                hdf[attr] = value
            elif value is None:
                hdf[attr] = 'None'
            elif isinstance(value, dict) is True:
                save_dict_to_hdf5(value, attr, hdf)


def load_class_from_hdf5(self, filename, ext='.h5', folder=None):
    """
    Loads Class attributes form HDF5 file.

    Parameters
    ----------
    self : Class
        Python Class object.
    filename : string
        Input filename.
    ext : string, optional
        Input extension.
    folder : None or string, optional
        Input folder. If 'None' is passed the current folder is used.

    Returns
    -------
    Nothing.
    """
    if folder is not None:
        infile = folder + filename + ext
    else:
        infile = filename + ext

    with h5py.File(infile, "r") as hdf:
        for key in hdf.keys():
            if isinstance(hdf[key], h5py.Dataset):
                item = parse_dataset_item(hdf[key])
            else:
                item = load_dict_from_hdf5(hdf, key)
            setattr(self, key, item)
