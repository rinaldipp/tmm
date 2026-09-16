"""
HDF5 persistence helpers.

This module stores Python object attributes and nested dictionaries in HDF5 files. The helpers are used by
``TMM.save()`` and ``TMM.load()`` as a package-internal checkpoint format.

For further information check the function specific documentation.
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
    Save a dictionary into an open HDF5 file as a group.

    Parameters
    ----------
    dic : dict
        Dictionary to store, possibly nested.
    key : string
        Group name.
    h5file : h5py.File
        Output file, already open.
    """
    group = h5file.create_group(_encode_key(key))
    recursively_save_dict_contents_to_group(h5file, group.name + "/", dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Write a dictionary's items under an HDF5 group, recursing into nested dictionaries.

    Parameters
    ----------
    h5file : h5py.File
        Output file, already open.
    path : string
        Group path.
    dic : dict
        Dictionary to store.
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
    Load a dictionary from an open HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Input file, already open.
    key : string
        Group name.

    Returns
    -------
    Dictionary, possibly nested.
    """
    return recursively_load_dict_contents_from_group(h5file, _encode_key(key) + "/")


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Read an HDF5 group back into a dictionary, recursing into subgroups.

    Parameters
    ----------
    h5file : h5py.File
        Input file, already open.
    path : string
        Group path.

    Returns
    -------
    Dictionary with the group's contents.
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
    Convert an HDF5 dataset to a Python or NumPy value according to its stored type.

    Parameters
    ----------
    item : h5py.Dataset
        Dataset to convert.

    Returns
    -------
    Scalar, list or array, depending on the stored type.
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
    Save an object's attributes into an HDF5 file.

    Parameters
    ----------
    self : object
        Object whose attributes are stored.
    filename : string, optional
        Output filename.
    ext : string, optional
        Output extension.
    folder : None or string, optional
        Output folder. If ``None``, the current working directory is used.
    timestamp : bool, optional
        If True, prefix the filename with a timestamp.
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
    Load attributes from an HDF5 file onto an object.

    Parameters
    ----------
    self : object
        Object that receives the attributes.
    filename : string
        Input filename.
    ext : string, optional
        Input extension.
    folder : None or string, optional
        Input folder. If ``None``, the current working directory is used.
    """
    infile = _hdf5_path(filename, ext=ext, folder=folder, timestamp=False)

    with h5py.File(infile, "r") as hdf:
        for key in hdf.keys():
            if isinstance(hdf[key], h5py.Dataset):
                item = parse_dataset_item(hdf[key])
            else:
                item = load_dict_from_hdf5(hdf, key)
            setattr(self, key, item)
