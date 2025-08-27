import glob
import os
import xarray as xr
from collections import namedtuple
import xmitgcm as xm


def fix_dimension(ds, bad_dim, new_dim, trim=True):
    assert len(ds[bad_dim]) == (len(ds[new_dim]) + 1)
    if trim and (len(ds[bad_dim]) == (len(ds[new_dim]) + 1)):
        ds = ds.isel(**{bad_dim: slice(0,-1)})
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

def open_mncdataset(fname_base, ntiles_y, ntiles_x, iternum=None):
    if iternum is not None:
        itersuf = '.%010d' % iternum
    else:
        flist = glob.glob(fname_base + "*.nc")
        flist = [os.path.basename(f) for f in flist]
        itersuf = '.%010d' % int(flist[0].split('.')[1])
    dsets_y = []
    for ny in range(ntiles_y):
        dsets_x = []
        swap_vars = set()
        for nx in range(ntiles_x):
            ntile = nx + ntiles_x*ny + 1
            fname = fname_base + '%s.t%03d.nc' % (itersuf, ntile)
            ds = xr.open_dataset(fname)
            ds, swapped_vars_x = fix_dimension(ds, 'Xp1', 'X')    
            ds = ds.chunk()
            dsets_x.append(ds)
        ds_xconcat = xr.concat(dsets_x, 'X')
        ds_xconcat, swapped_vars_y = fix_dimension(ds_xconcat, 'Yp1', 'Y')
        dsets_y.append(ds_xconcat)
    ds = xr.concat(dsets_y, 'Y')
    ds = reset_dimensions(ds, 'Xp1', 'X', *swapped_vars_x)
    ds = reset_dimensions(ds, 'Yp1', 'Y', *swapped_vars_y)

    return ds


SwirlInputData = namedtuple('SwirlInputData', [
    'ds_mitgcm', 'times', 'depths', 'time_indices', 'depth_indices',
    'dx', 'dy', 'dz_array',
    'uvel_data', 'vvel_data', 'wvel_data', 'theta_data'
])


# Extract and compute numpy arrays BEFORE passing to dask (improves reading data time)
def load_input_data_netcdf(mitgcm_nc_results_path, output_folder, output_grid_folder_name, px, py, time_indices=None, depth_indices=None):
    """
    :return: namedtuple('SwirlInputData', [
    'ds_mitgcm', 'times', 'depths', 'time_indices', 'depth_indices',
    'dx', 'dy', 'dz_array',
    'uvel_data', 'vvel_data', 'wvel_data', 'theta_data'
    ])
    """
    ds_mitgcm = open_mncdataset(os.path.join(mitgcm_nc_results_path, '3Dsnaps'), py, px)

    ds_grid = xr.open_dataset(str(os.path.join(output_folder, output_grid_folder_name, 'merged_grid.nc')))

    times = ds_mitgcm.T.values
    depths = ds_grid.Z.values

    if time_indices == None:
        time_indices = range(len(times))
    if depth_indices == None:
        depth_indices = range(len(depths))

    dx = ds_grid.dxC.values[0][0]
    dy = ds_grid.dyC.values[0][0]
    dz_array = ds_grid.drC.values
    uvel_data = ds_mitgcm['UVEL'].isel(T=time_indices, Zmd000100=depth_indices).fillna(0).values
    vvel_data = ds_mitgcm['VVEL'].isel(T=time_indices, Zmd000100=depth_indices).fillna(0).values
    wvel_data = ds_mitgcm['WVEL'].isel(T=time_indices, Zld000100=depth_indices).fillna(0).values
    theta_data = ds_mitgcm['THETA'].isel(T=time_indices, Zmd000100=depth_indices).fillna(0).values

    return SwirlInputData(ds_mitgcm,
                          times, depths,
                          time_indices, depth_indices,
                          dx, dy, dz_array,
                          uvel_data, vvel_data, wvel_data, theta_data)


# Extract and compute numpy arrays BEFORE passing to dask (improves reading data time)
def load_input_data_binary(mitgcm_bin_results_path, binary_mitgcm_grid_folder_path, ref_date, dt_mitgcm_results,
                           time_indices=None, depth_indices=None):
    ds_mitgcm = xm.open_mdsdataset(
                                mitgcm_bin_results_path,
                                grid_dir=binary_mitgcm_grid_folder_path,
                                ref_date=ref_date,
                                prefix='3Dsnaps',
                                delta_t=dt_mitgcm_results,
                                endian=">")

    times = ds_mitgcm.time.values
    depths = ds_mitgcm.Z.values

    if time_indices == None:
        time_indices = range(len(times))
    if depth_indices == None:
        depth_indices = range(len(depths))

    dx = ds_mitgcm.dxC.values[0][0]
    dy = ds_mitgcm.dyC.values[0][0]
    dz_array = ds_mitgcm.drC.values
    uvel_data = ds_mitgcm['UVEL'].isel(time=time_indices, Z=depth_indices).fillna(0).values
    vvel_data = ds_mitgcm['VVEL'].isel(time=time_indices, Z=depth_indices).fillna(0).values
    wvel_data = ds_mitgcm['WVEL'].isel(time=time_indices, Zl=depth_indices).fillna(0).values
    theta_data = ds_mitgcm['THETA'].isel(time=time_indices, Z=depth_indices).fillna(0).values

    return SwirlInputData(ds_mitgcm,
                          times, depths,
                          time_indices,depth_indices,
                          dx, dy, dz_array,
                          uvel_data, vvel_data, wvel_data, theta_data)
