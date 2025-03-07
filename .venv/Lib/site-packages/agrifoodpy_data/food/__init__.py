"""Food module"""

import os
import xarray as xr

available = ['FAOSTAT', "Nutrients_FAOSTAT"]
data_dir = os.path.join(os.path.dirname(__file__), 'data/' )

def __getattr__(name):
    if name not in available:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")

    _data_file = f'{data_dir}{name}.nc'

    # If the file contains more than a single dataarray, then it will try
    # to load it as a dataset
    try:
        with xr.open_dataset(_data_file) as data:
            data.load()
            return data
        
    except ValueError:
        with xr.open_dataarray(_data_file) as data:
            data.load()
            return data