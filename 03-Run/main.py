"""
Created on 30.07.2025 by Anne Leroquais (EAWAG)
Here, we use netcdf results from MITgcm to detect eddies in lakes.
"""

import pandas as pd
import os
import time
from datetime import datetime
from collections import namedtuple

import xarray as xr

from functions.run_swirl import run_swirl
from functions.merge_nc_results import open_mncdataset
from functions.create_lvl0 import compute_ke_snapshot, extract_eddy_data

import dask
from dask import delayed, compute


I_MITGCM_FOLDER_PATH = r"../run" # Folder containing tiled mitgcm results
SWIRL_PARAMS_NAME = 'swirl_03'
MODEL = 'geneva_200m'
OUTPUT_FOLDER = r'../output'

O_MITGCM_FOLDER_NAME = 'mitgcm_results'
O_GRID_FOLDER_NAME = 'grid'
O_LVL0_FOLDER_NAME = 'eddy_catalogues_lvl0'

SwirlInputData = namedtuple('SwirlInputData', [
    'ds_mitgcm', 'times', 'depths', 'time_indices', 'depth_indices',
    'dx', 'dy', 'dz_array',
    'uvel_data', 'vvel_data', 'wvel_data', 'theta_data'
])


# Extract and compute numpy arrays BEFORE passing to dask (improves reading data time)
def load_input_data(mitgcm_nc_results_path, time_indices=None, depth_indices=None):
    ds_mitgcm = open_mncdataset(os.path.join(mitgcm_nc_results_path, '3Dsnaps'), 12, 48)
    ds_grid = xr.open_dataset(os.path.join(OUTPUT_FOLDER, O_GRID_FOLDER_NAME, 'merged_grid.nc'))

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
                          time_indices,depth_indices,
                          dx, dy, dz_array,
                          uvel_data, vvel_data, wvel_data, theta_data)


def run_swirl_and_create_lvl0(uvel, vvel, wvel, theta,
                              dx, dy, dz,
                              swirl_params_file,
                              date, depth,
                              t_index, d_index,
                              id_level0=0):
    eddies = run_swirl(uvel, vvel, dx, dy, swirl_params_file)
    if not eddies:  # empty list
        return pd.DataFrame()  # optionally: return with predefined columns

    ke_grid = compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz)
    eddy_rows = []
    for eddy_index in range(len(eddies)):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(indices_eddy, eddies[eddy_index], date, depth, dz, ke_grid, dx*dy, theta, id_level0)
        eddy_rows.append(row_data)

    return pd.concat([pd.DataFrame([row]) for row in eddy_rows], ignore_index=True)


def get_str_current_time():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def main():
    dask.config.set(scheduler='threads')
    #---------------------------------
    print(f'Loading input data... ({get_str_current_time()})')
    swirl_input_data = load_input_data(I_MITGCM_FOLDER_PATH)
    start_date_str = pd.Timestamp(swirl_input_data.ds_mitgcm.T.values[0]).strftime('%Y%m%d')
    end_date_str = pd.Timestamp(swirl_input_data.ds_mitgcm.T.values[-1]).strftime('%Y%m%d')

    # ---------------------------------
    print(f'Saving merged mitgcm results... ({get_str_current_time()})')
    swirl_input_data.ds_mitgcm.to_zarr(os.path.join(OUTPUT_FOLDER, O_MITGCM_FOLDER_NAME, rf"mitgcm_{start_date_str}_{end_date_str}.zarr"), 
                        mode="w", 
                        compute=True, 
                        consolidated=False)

    # ---------------------------------
    print(f'Detecting eddies and creating level 0 catalogue... ({get_str_current_time()})')
    dask.config.set(scheduler='processes', num_workers=576)
    tasks = {}
    for ti, t_idx in enumerate(swirl_input_data.time_indices):
        date = pd.Timestamp(swirl_input_data.times[ti]).to_pydatetime()
        for di, d_idx in enumerate(swirl_input_data.depth_indices):
            depth = float(swirl_input_data.depths[di])
            dz = swirl_input_data.dz_array[di]
            uvel = swirl_input_data.uvel_data[ti, di].T
            vvel = swirl_input_data.vvel_data[ti, di].T
            wvel = swirl_input_data.wvel_data[ti, di].T
            theta = swirl_input_data.theta_data[ti, di].T
            tasks[(t_idx, d_idx)] = delayed(run_swirl_and_create_lvl0)(uvel, vvel, wvel, theta,
                                                                       swirl_input_data.dx, swirl_input_data.dy, dz,
                                                                       SWIRL_PARAMS_NAME,
                                                                       date, depth,
                                                                       ti, di)

    # Compute all tasks in parallel
    results = compute(*tasks.values())

    # Create final DataFrame
    df_catalogue_level0 = pd.concat([row for row in results], ignore_index=True)

    # ---------------------------------
    output_path = os.path.join(OUTPUT_FOLDER, O_LVL0_FOLDER_NAME, f'lvl0_{start_date_str}_{end_date_str}.csv')
    print(f'Saving catalogue level 0 to {output_path}... ({get_str_current_time()})')
    df_catalogue_level0.to_csv(output_path, index=False)
    
    print(f'Done. ({get_str_current_time()})')


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # optional on Linux/macOS
    main()