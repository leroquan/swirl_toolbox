"""
Created on 30.07.2025 by Anne Leroquais (EAWAG) - memory-safe dask version

Notes:
- Expects load_input_data_netcdf / load_input_data_binary to return lazy xarray DataArrays
  chunked so that .isel(time=..., Z=...) returns a small Dask graph (1 plane).
"""

import os
import time
from datetime import datetime
import json

import pandas as pd
import numpy as np

import dask
from dask import delayed, compute

import matplotlib.pyplot as plt

from functions import run_swirl, plot_map_swirl
from functions import load_input_data_netcdf, load_input_data_binary
from functions import reformat_mitgcm_results
from functions import compute_ke_snapshot, extract_eddy_data
from functions import save_to_netcdf

O_MITGCM_FOLDER_NAME = 'mitgcm_results'
O_GRID_FOLDER_NAME = 'grid'
O_LVL0_FOLDER_NAME = 'eddy_catalogues_lvl0'


def get_str_current_time():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def run_parallel_task(ds_sel,
                      dx, dy, dz,
                      swirl_params_path,
                      date, depth,
                      t_index, d_index,
                      output_folder,
                      id_level0=0):
    """
    u_da, v_da, w_da, th_da: xarray.DataArray objects (lazy/dask-backed) for a single (T,Z) plane
    This function *computes* those slices locally (inside the worker), so memory usage stays bounded.
    """
    # Compute the 2D arrays inside the worker. Transpose to match your previous orientation if needed.
    # We call .compute() on the DataArray which returns a NumPy array (or xarray.DataArray) depending on your backend.
    print(f'Loading input data from t={t_index}, d={d_index}')
    u = ds_sel.u.T.values
    v = ds_sel.v.T.values
    w = ds_sel.w.T.values
    theta = ds_sel.t.T.values

    # Eddy detection & feature extraction (same as before)
    print(f'Run swirl for t={t_index}, d={d_index}')
    eddies = run_swirl(u, v, dx, dy, swirl_params_path)
    if not eddies:
        return pd.DataFrame()  # empty result for this slice

    print(f'Compute KE for t={t_index}, d={d_index}')
    ke_grid = compute_ke_snapshot(u, v, w, dx, dy, dz)

    print(f'Extract eddy data for t={t_index}, d={d_index}')
    eddy_rows = []
    for eddy_index, eddy in enumerate(eddies):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(indices_eddy, eddy, date, depth, dz, ke_grid, dx*dy, theta, id_level0)
        eddy_rows.append(row_data)

    # Optional map output for first depth
    if d_index == 0:
        print(f'Map t={t_index}, d={d_index}')
        fig = plot_map_swirl(u.T, v.T, eddies, date, 6)
        out_fig_dir = os.path.join(output_folder, 'figures')
        os.makedirs(out_fig_dir, exist_ok=True)
        fig_path = os.path.join(out_fig_dir, f'map_swirl_{date}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # IMPORTANT: free matplotlib memory

    print(f'Concatenating t={t_index}, d={d_index}')
    return pd.concat([pd.DataFrame([r]) for r in eddy_rows], ignore_index=True)


def main(config_path='..//postprocessing//config_postprocessing.json'):
    with open(config_path, 'r') as f:
        pp_config = json.load(f)

    nb_cores = int(pp_config.get('px', 1)) * int(pp_config.get('py', 1))
    dask.config.set(scheduler='processes', num_workers=nb_cores)

    # ---------------------------------
    print(f'Loading input data... ({get_str_current_time()})')
    if pp_config['mitgcm_output_format'] == 'netcdf':
        swirl_input_data = load_input_data_netcdf(pp_config['i_mitgcm_folder_path'],
                                                  pp_config['output_folder'],
                                                  O_GRID_FOLDER_NAME,
                                                  pp_config['px'], pp_config['py'])
    elif pp_config['mitgcm_output_format'] == 'binary':
        swirl_input_data = load_input_data_binary(pp_config['i_mitgcm_folder_path'],
                                                  pp_config['binary_mitgcm_grid_folder_path'],
                                                  pp_config['binary_ref_date'],
                                                  pp_config['binary_dt'])
    else:
        raise ValueError(f'"mitgcm_output_format" must be either "netcdf" or "binary": {pp_config["mitgcm_output_format"]}')

    ds_reformat = reformat_mitgcm_results(swirl_input_data.ds_mitgcm,
                                          swirl_input_data.times, swirl_input_data.depths,
                                          pp_config['grid_folder'],
                                          nodata=np.nan)

    start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime('%Y%m%d')
    end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime('%Y%m%d')

    # --------------------------------
    # Optionally save merged MITgcm results (be cautious with memory)
    if str(pp_config.get('save_nc_mitgcm', "False")) == "True":
        print(f'Saving merged mitgcm results... ({get_str_current_time()})')
        # Saves the dataset in chunks (avoids loading entire dataset in memory)
        output_path = os.path.join(pp_config['output_folder'],
                                   O_MITGCM_FOLDER_NAME,
                                   f"mitgcm_{start_date_str}_{end_date_str}.nc")
        save_to_netcdf(ds_reformat, output_path)

    # ---------------------------------
    print(f'Preparing parallel tasks... ({get_str_current_time()})')
    delayed_tasks = {}
    ds = ds_reformat.fillna(0)
    dates = pd.to_datetime(swirl_input_data.times).to_pydatetime()
    for ti, time_val in enumerate(swirl_input_data.times):
        date = dates[ti]
        for di, depth_val in enumerate(swirl_input_data.depths):
            depth = float(depth_val)
            dz = swirl_input_data.dz_array[di]

            delayed_tasks[(ti, di)] =(
                delayed(run_parallel_task)(
                    delayed(ds.isel)(time=ti, depth=di),
                    swirl_input_data.dx, swirl_input_data.dy, dz,
                    pp_config['swirl_params_path'],
                    date, depth,
                    ti, di,
                    pp_config['output_folder']
                )
            )

    # ---------------------------------
    print(f'Computing parallel tasks... ({get_str_current_time()})')
    # Convert dict to list of (key, task)
    task_items = list(delayed_tasks.items())

    results = {}
    for i in range(0, len(task_items), nb_cores):
        batch = task_items[i:i + nb_cores]
        keys, tasks = zip(*batch)
        batch_results = dask.compute(*tasks)
        for k, r in zip(keys, batch_results):
            results[k] = r

    # ---------------------------------
    print(f'Finished all tasks. Concatenating and saving to csv... ({get_str_current_time()})')
    df_catalogue_level0 = pd.concat([row for row in results], ignore_index=True)

    # ---------------------------------
    # Prepare output folder + CSV
    out_dir = os.path.join(pp_config['output_folder'], O_LVL0_FOLDER_NAME)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'lvl0_{start_date_str}_{end_date_str}.csv')
    df_catalogue_level0.to_csv(output_path, index=False)

    # ---------------------------------
    print(f'Checking if MITgcm diverged... ({get_str_current_time()})')
    last_theta_slice = swirl_input_data.theta_data.isel(time=-1, Z=0).values
    if np.all(np.isnan(last_theta_slice)):
        raise ValueError('Last time step of MITgcm results contains only NaN values.')

    # ---------------------------------
    print(f'Done. ({get_str_current_time()})')


if __name__ == "__main__":
    # multiprocessing compatibility for Windows (harmless on Linux)
    from multiprocessing import freeze_support
    freeze_support()
    main()