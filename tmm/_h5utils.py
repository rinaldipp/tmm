"""
HDF5 persistence helpers.

This module stores Python object attributes and nested dictionaries in HDF5
files. The helpers are used by ``TMM.save()`` and ``TMM.load()`` as a
package-internal checkpoint format.
"""

from pathlib import Path
import time

import h5py
import numpy as np


def _encode_key(key):
    """Return a HDF5-safe representation of a dictionary key."""
    return str(key).replace("/", "_div_")


def _decode_key(key):
    """Restore a dictionary key without executing arbitrary text."""
    decoded = key.replace("_div_", "/")
    try:
        integer = int(decoded)
    except ValueError:
        return decoded
    if str(integer) == decoded:
        return integer
    return decoded


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
    """
    group = h5file.create_group(_encode_key(key))
    recursively_save_dict_contents_to_group(h5file, group.name + "/", dic)


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
    """
    for key, item in dic.items():
        encoded_key = _encode_key(key)
        if isinstance(item, dict):
            group_path = path + encoded_key
            h5file.create_group(group_path)
            recursively_save_dict_contents_to_group(h5file, group_path + "/", item)
        else:
            h5file[path + encoded_key] = item if item is not None else "None"


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
    Dictionary that can contain other dictionaries inside.
    """
    return recursively_load_dict_contents_from_group(h5file, _encode_key(key) + "/")


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
    Dictionary containing the values inside the h5 group.
    """
    ans = {}
    for key, item in h5file[path].items():
        dict_key = _decode_key(key)
        if isinstance(item, h5py.Dataset):
            ans[dict_key] = parse_dataset_item(item)
        elif isinstance(item, h5py.Group):
            ans[dict_key] = recursively_load_dict_contents_from_group(h5file, path + key + "/")
    return ans


def _parse_scalar(value):
    """Return a Python scalar for scalar HDF5 values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("UTF-8")
    if isinstance(value, str):
        return None if value == "None" else value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def parse_dataset_item(item):
    """
    Parses a HDF5 Dataset based on the datatype.

    Parameters
    ----------
    item : h5py.Dataset
        Dataset containing data that will be parsed.

    Returns
    -------
    Parsed dataset.
    """
    value = item[()]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _parse_scalar(value.item())
        if np.issubdtype(value.dtype, np.bool_):
            return value.astype(bool).tolist()
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(int).tolist()
        return value
    return _parse_scalar(value)


def _hdf5_path(filename, ext=".h5", folder=None, timestamp=False):
    """Return the output or input path for a HDF5 file."""
    base = Path(folder) if folder is not None else Path.cwd()
    prefix = time.strftime("%Y%m%d-%H%M_") if timestamp else ""
    return base / f"{prefix}{filename}{ext}"


def save_class_to_hdf5(self, filename="class", ext=".h5", folder=None, timestamp=False):
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
    """
    outfile = _hdf5_path(filename, ext=ext, folder=folder, timestamp=timestamp)

    with h5py.File(outfile, "w") as hdf:
        for attr, value in vars(self).items():
            if isinstance(value, dict):
                save_dict_to_hdf5(value, attr, hdf)
            else:
                hdf[attr] = value if value is not None else "None"


def load_class_from_hdf5(self, filename, ext=".h5", folder=None):
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
    """
    infile = _hdf5_path(filename, ext=ext, folder=folder, timestamp=False)

    with h5py.File(infile, "r") as hdf:
        for key in hdf.keys():
            if isinstance(hdf[key], h5py.Dataset):
                item = parse_dataset_item(hdf[key])
            else:
                item = load_dict_from_hdf5(hdf, key)
            setattr(self, key, item)
