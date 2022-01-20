import xarray as xr
import numpy as np
from typing import Union
import json


def open_feltordataset(
    datapath: str = "./*.nc",
    chunks: Union[int, dict] = None,
    restart_indices: bool = False,
    probes: bool = False,
    probe_id: str = "Probes",
    probex: str = "px",
    probey: str = "py",
    **kwargs: dict,
) -> xr.Dataset:
    """Loads FELTOR output into one xarray Dataset. Can load either a single
    output file or multiple coherent files for restarted simulations.

    Parameters
    ----------
    datapath : str or (list or tuple of xr.Dataset), optional
        Path to the data to open. Can point to either a set of one or more *nc
        files.
    chunks : dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        By default, chunks will be chosen to load entire input files into memory at once.
        This has a major impact on performance: please see the full documentation for more details:
        http://xarray.pydata.org/en/stable/user-guide/dask.html#chunking-and-performance
    restart_indices: bool, optional
        if True, duplicate time steps from restared runs are kept
    probes: bool, optional
        if True, indicates that the dataset contains probes and associates values of the
        x and y possition for each probe with the coresponding probe_id.
        Also changes the combine option to "by_coords".
    probe_id: str, optional
        The coordinate to associate the x and y positions of the probes with.
    probex: str, optional
        The name of the variable in the dataset where the x positon of probes are stored.
    probey: str, optional
        The name of the variable in the dataset where the y positon of probes are stored.
    kwargs : optional
        Keyword arguments are passed down to `xarray.open_mfdataset`, which in
        turn passes extra kwargs down to `xarray.open_dataset`.
    """
    if chunks is None:
        chunks = {}

    if probes is True:
        combine_opt = "by_coords"
    else:
        combine_opt = "nested"

    ds = xr.open_mfdataset(
        datapath,
        chunks=chunks,
        combine=combine_opt,
        concat_dim="time",
        decode_times=False,
        join="outer",
        **kwargs,
    )

    if restart_indices:
        return ds

    _, index = np.unique(ds["time"], return_index=True)

    # store inputfile data in ds.attrs
    input_variables = json.loads(ds.attrs["inputfile"])

    for i in input_variables:
        ds.attrs[i] = input_variables[i]

    if probes is True:
        ds = ds.assign_coords(dict(probex=(probe_id,ds[probex].values),probey=(probe_id,ds[probey].values)))

    return ds.isel(time=index)
