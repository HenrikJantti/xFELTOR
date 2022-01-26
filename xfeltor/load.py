import xarray as xr
import numpy as np
from typing import Union
import json


def open_feltordataset(
    datapath: str = "./*.nc",
    chunks: Union[int, dict] = None,
    restart_indices: bool = False,
    probes: bool = False,
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
        Keyword arguments are passed down to `xarray.open_mfdataset`, which in
        turn passes extra kwargs down to `xarray.open_dataset`.
    probes: bool, optional
        if True, indicates that the dataset contains probes and associates values of the
        x and y possition for each probe with the coresponding probe_id.
        Also changes the combine option to "by_coords".
    """
    if chunks is None:
        chunks = {}

    combine_opt = "by_coords" if probes else "nested"

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

    if probes:
        x = np.unique(ds.px.values)
        y = np.unique(ds.py.values)
        ds = ds.assign_coords(
            dict(
                probe_x=x,
                probe_y=y,
            )
        )
        reshaped_prb = np.reshape(
            ds.electrons_prb.values, (y.size, x.size, ds.probe_time.values.size)
        )
        ds = ds.assign(
            electrons_prb=(["probe_y", "probe_x", "probe_time"], reshaped_prb)
        )
        reshaped_prb = np.reshape(
            ds.ions_prb.values, (y.size, x.size, ds.probe_time.values.size)
        )
        ds = ds.assign(ions_prb=(["probe_y", "probe_x", "probe_time"], reshaped_prb))
        reshaped_prb = np.reshape(
            ds.potential_prb.values, (y.size, x.size, ds.probe_time.values.size)
        )
        ds = ds.assign(
            potential_prb=(["probe_y", "probe_x", "probe_time"], reshaped_prb)
        )
        reshaped_prb = np.reshape(
            ds.vorticity_prb.values, (y.size, x.size, ds.probe_time.values.size)
        )
        ds = ds.assign(
            vorticity_prb=(["probe_y", "probe_x", "probe_time"], reshaped_prb)
        )
        ds = ds.drop_dims(("probes"))

    return ds.isel(time=index)
