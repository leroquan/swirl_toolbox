import glob
import os
import numpy as np
import pandas as pd
import sys
import xarray as xr
import json
from collections import namedtuple
import xmitgcm as xm


SwirlInputData = namedtuple(
    "SwirlInputData", ["ds_mitgcm", "times", "depths", "dx", "dy", "dz_array"]
)


class MitgcmGrid:
    """Class representing an MITgcm grid, with optional loading from .npy files."""

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.lat_grid = np.array([])
        self.lon_grid = np.array([])
        self.dz = np.array([])
        self.parameters = {}

    def load_from_path(self, path_grid: str):
        try:
            self.x = np.load(os.path.join(path_grid, "x.npy"))
            self.y = np.load(os.path.join(path_grid, "y.npy"))
            self.lat_grid = np.load(os.path.join(path_grid, "lat_grid.npy"))
            self.lon_grid = np.load(os.path.join(path_grid, "lon_grid.npy"))
            self.dz = pd.read_csv(
                os.path.join(path_grid, "dz.csv"), header=None
            ).to_numpy()
            with open(os.path.join(path_grid, "parameters.json"), "r") as file:
                self.parameters = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing grid file: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading grid data: {e}") from e


def get_mitgcm_grid(path_folder_grid: str) -> MitgcmGrid:
    grid = MitgcmGrid()
    grid.load_from_path(path_folder_grid)
    return grid


def fix_dimension(ds, bad_dim, new_dim, trim=True):
    assert len(ds[bad_dim]) == (len(ds[new_dim]) + 1)
    if trim and (len(ds[bad_dim]) == (len(ds[new_dim]) + 1)):
        ds = ds.isel(**{bad_dim: slice(0, -1)})
        assert len(ds[bad_dim]) == len(ds[new_dim])
    swapped_vars = []
    for v in ds:
        # replace naughty dimension
        dims = list(ds[v].dims)
        if bad_dim in dims:
            idx = dims.index(bad_dim)
            dims[idx] = new_dim
            ds[v].variable.dims = dims
            swapped_vars.append(v)
    return ds, swapped_vars


def reset_dimensions(ds, orig_dim, new_dim, *reset_vars):
    for v in reset_vars:
        dims = list(ds[v].dims)
        if new_dim in dims:
            idx = dims.index(new_dim)
            dims[idx] = orig_dim
            ds[v].variable.dims = dims
    # reindexing necessary to figure out new dims
    return ds.reindex()


def bk_open_mncdataset(fname_base, ntiles_y, ntiles_x, iternum=None):
    if iternum is not None:
        itersuf = ".%010d" % iternum
    else:
        flist = glob.glob(fname_base + "*.nc")
        flist = [os.path.basename(f) for f in flist]
        itersuf = ".%010d" % int(flist[0].split(".")[1])
    dsets_y = []
    for ny in range(ntiles_y):
        dsets_x = []
        swap_vars = set()
        for nx in range(ntiles_x):
            ntile = nx + ntiles_x * ny + 1
            fname = fname_base + "%s.t%03d.nc" % (itersuf, ntile)
            ds = xr.open_dataset(fname)
            ds, swapped_vars_x = fix_dimension(ds, "Xp1", "X")
            ds = ds.chunk()
            dsets_x.append(ds)
        ds_xconcat = xr.concat(dsets_x, "X")
        ds_xconcat, swapped_vars_y = fix_dimension(ds_xconcat, "Yp1", "Y")
        dsets_y.append(ds_xconcat)
    ds = xr.concat(dsets_y, "Y")
    ds = reset_dimensions(ds, "Xp1", "X", *swapped_vars_x)
    ds = reset_dimensions(ds, "Yp1", "Y", *swapped_vars_y)

    return ds


def open_mncdataset(fname_base, ntiles_y, ntiles_x, iternum=None):
    if iternum is not None:
        itersuf = f".{iternum:010d}"
    else:
        flist = glob.glob(fname_base + "*.nc")
        flist = [os.path.basename(f) for f in flist]
        itersuf = f".{int(flist[0].split('.')[1]):010d}"

    # Build full list of tile files
    fnames = []
    for ny in range(ntiles_y):
        row = []
        for nx in range(ntiles_x):
            ntile = nx + ntiles_x * ny + 1
            fname = fname_base + f"{itersuf}.t{ntile:03d}.nc"
            row.append(fname)
        fnames.append(row)  # <-- make it 2D (list of lists)

    # Open all files at once
    ds = xr.open_mfdataset(
        fnames,
        concat_dim=[
            "Y",
            "X",
        ],  # <-- may need adjustment depending on how your dims are labeled
        combine="nested",  # combine based on position in list
        chunks={"time": 1, "Z": -1, "Zl": -1},  # let user pick Dask chunking
        engine="netcdf4",
        parallel=False,
    )

    return ds


def _standardize_dims(ds, kind="netcdf"):
    """
    Rename dims/coords so both code paths expose:
      T (time), Z (mid), Zl (lower), Y, X
    and variables UVEL/VVEL use (T,Z,Y,X), WVEL uses (T,Zl,Y,X), THETA uses (T,Z,Y,X).
    """
    if kind == "netcdf":
        # common MITgcm NetCDF export names from your snippet
        ren = {
            "T": "time",
            "Zmd000100": "Z",  # mid
            "Zld000100": "Zl",  # lower
        }
        # already OK for Y, X after open_mncdataset
        ds = ds.rename({k: v for k, v in ren.items() if k in ds.dims})
    else:  # binary (xmitgcm)
        # xmitgcm default dims are usually: time, Z, Zl, YC, XC
        ren = {}
        # if 'time' in ds.dims: ren['time'] = 'T'
        # Z and Zl already match desired names
        ds = ds.rename(ren)

    return ds


def _ensure_native_endian(mitgcm_ds):
    # Determine native endianness
    native_endian = "<" if sys.byteorder == "little" else ">"

    # Check a sample variable
    sample_var = list(mitgcm_ds.data_vars)[0]
    current_dtype = mitgcm_ds[sample_var].dtype

    # Check if conversion is needed
    # dtype.byteorder can be: '<' (little), '>' (big), '=' (native), '|' (not applicable)
    if current_dtype.byteorder in (native_endian, "=", "|"):
        # Already native or byte-order doesn't matter
        return mitgcm_ds

    # Only convert if needed
    return mitgcm_ds.astype(f"{native_endian}f8")


def load_input_data_netcdf(
    mitgcm_nc_results_path, grid_folder_path, px, py, endian=">"
):
    ds_mitgcm = open_mncdataset(os.path.join(mitgcm_nc_results_path, "3Dsnaps"), py, px)
    ds_mitgcm = _standardize_dims(ds_mitgcm, kind="netcdf").chunk(
        {"time": 1, "Z": -1, "Zl": -1}
    )
    if endian == ">":
        ds_mitgcm = _ensure_native_endian(ds_mitgcm)

    ds_grid = xr.open_dataset(grid_folder_path)

    times = ds_mitgcm["time"].values
    depths = ds_grid["Z"].values

    dx = float(ds_grid.dxC.values[0][0])
    dy = float(ds_grid.dyC.values[0][0])
    dz_array = ds_grid.drC.values

    return SwirlInputData(ds_mitgcm, times, depths, dx, dy, dz_array)


def load_input_data_binary(
    mitgcm_bin_results_path,
    binary_mitgcm_grid_folder_path,
    ref_date,
    dt_mitgcm_results,
    endian=">",
):
    ds_mitgcm = xm.open_mdsdataset(
        mitgcm_bin_results_path,
        grid_dir=binary_mitgcm_grid_folder_path,
        ref_date=ref_date,
        prefix="3Dsnaps",
        delta_t=dt_mitgcm_results,
        endian=endian,
        iters="all",
    )

    ds_mitgcm = _standardize_dims(ds_mitgcm, kind="binary").chunk(
        {"time": 1, "Z": 10, "Zl": 10}
    )
    if endian == ">":
        ds_mitgcm = _ensure_native_endian(ds_mitgcm)

    times = ds_mitgcm["time"].values
    depths = ds_mitgcm["Z"].values

    dx = float(ds_mitgcm.dxC.values[0][0])
    dy = float(ds_mitgcm.dyC.values[0][0])
    dz_array = ds_mitgcm.drC.values

    return SwirlInputData(ds_mitgcm, times, depths, dx, dy, dz_array)
