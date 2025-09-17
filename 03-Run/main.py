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


def run_parallel_task(u, v, w, theta,
                      dx, dy, dz,
                      swirl_params_path,
                      date, depth,
                      t_index, d_index,
                      output_folder,
                      id_level0=0):

    # Eddy detection & feature extraction (same as before)
    print(f'Run swirl for t={t_index}, d={d_index}')
    eddies = run_swirl(u, v, dx, dy, swirl_params_path)
    if not eddies:
        return pd.DataFrame()  # empty result for this slice

    #print(f'Compute KE for t={t_index}, d={d_index}')
    ke_grid = compute_ke_snapshot(u, v, w, dx, dy, dz)

    #print(f'Extract eddy data for t={t_index}, d={d_index}')
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

    #print(f'Concatenating t={t_index}, d={d_index}')
    return pd.concat([pd.DataFrame([r]) for r in eddy_rows], ignore_index=True)


def compute_lake_properties(swirl_input_data, uvel_data, vvel_data, wvel_data, theta_data):
    ke_lake = []
    for ti, date in enumerate(pd.to_datetime(swirl_input_data.times).to_pydatetime()):
        for di, depth_val in enumerate(swirl_input_data.depths):
            u = uvel_data[ti, di].T
            v = vvel_data[ti, di].T
            w = wvel_data[ti, di].T
            theta = theta_data[ti, di].T

            ke_slice = compute_ke_snapshot(u, v, w, swirl_input_data.dx, swirl_input_data.dy,
                                           dz=swirl_input_data.dz_array[di]).sum()
            mean_temperature = theta[theta > 0].mean()
            surface = len(theta[theta > 0]) * swirl_input_data.dx * swirl_input_data.dy
            volume = surface * swirl_input_data.dz_array[di]

            ke_lake.append({'time_index': ti, 'depth_index': di, 'date': date, 'depth_[m]': depth_val,
                            'surface_area_[m2]': surface, 'volume_slice_[m3]': volume,
                            'kinetic_energy_[MJ]': ke_slice, 'mean_temperature_lake_[°C]': mean_temperature})

    return ke_lake


def main(config_path='..//postprocessing//config_postprocessing.json'):
    with open(config_path, 'r') as f:
        pp_config = json.load(f)

    nb_cores = int(pp_config.get('px', 1)) * int(pp_config.get('py', 1))
    dask.config.set(scheduler='processes', num_workers=nb_cores)

    # ---------------------------------
    print(f'Loading input data... ({get_str_current_time()})')
    if pp_config['mitgcm_output_format'] == 'netcdf':
        grid_nc_path = os.path.join(pp_config['output_folder'],
                                    O_GRID_FOLDER_NAME,
                                    'merged_grid.nc')
        swirl_input_data = load_input_data_netcdf(pp_config['i_mitgcm_folder_path'],
                                                  grid_nc_path,
                                                  pp_config['px'], pp_config['py'])
    elif pp_config['mitgcm_output_format'] == 'binary':
        swirl_input_data = load_input_data_binary(pp_config['i_mitgcm_folder_path'],
                                                  pp_config['binary_mitgcm_grid_folder_path'],
                                                  pp_config['binary_ref_date'],
                                                  pp_config['binary_dt'])
    else:
        raise ValueError(f'"mitgcm_output_format" must be either "netcdf" or "binary": {pp_config["mitgcm_output_format"]}')

    start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime('%Y%m%d')
    end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime('%Y%m%d')

    # --------------------------------
    # Optionally save merged MITgcm results (be cautious with memory)
    if str(pp_config.get('save_nc_mitgcm', "False")) == "True":
        print(f'Reformatting input data... ({get_str_current_time()})')
        ds_reformat = reformat_mitgcm_results(swirl_input_data.ds_mitgcm,
                                              swirl_input_data.times, swirl_input_data.depths,
                                              pp_config['grid_folder'],
                                              nodata=np.nan)

        print(f'Saving merged mitgcm results... ({get_str_current_time()})')
        mitgcm_output_folder = os.path.join(pp_config['output_folder'], O_MITGCM_FOLDER_NAME)
        os.makedirs(mitgcm_output_folder, exist_ok=True)
        output_path = os.path.join(mitgcm_output_folder, f"mitgcm_{start_date_str}_{end_date_str}.nc")
        save_to_netcdf(ds_reformat, output_path)

    # ---------------------------------
    print(f'Loading data into RAM... ({get_str_current_time()})')
    ds = swirl_input_data.ds_mitgcm
    uvel_data = ds.UVEL.values
    vvel_data = ds.VVEL.values
    wvel_data = ds.WVEL.values
    theta_data = ds.THETA.values

    print(f'Computing lake properties... ({get_str_current_time()})')
    ke_lake = compute_lake_properties(swirl_input_data,
                                      uvel_data, vvel_data, wvel_data, theta_data)

    print(f'Saving lake properties... ({get_str_current_time()})')
    lvl0_out_dir = os.path.join(pp_config['output_folder'], O_LVL0_FOLDER_NAME)
    os.makedirs(lvl0_out_dir, exist_ok=True)
    output_path = os.path.join(lvl0_out_dir, f'lake_characteristics_{start_date_str}_{end_date_str}.csv')
    pd.DataFrame(ke_lake).to_csv(output_path, index=False)

    print(f'Preparing parallel tasks... ({get_str_current_time()})')
    delayed_tasks = {}
    for ti, date in enumerate(pd.to_datetime(swirl_input_data.times).to_pydatetime()):
        for di, depth_val in enumerate(swirl_input_data.depths):
            depth = float(depth_val)
            dz = swirl_input_data.dz_array[di]
            uvel = uvel_data[ti, di].T
            vvel = vvel_data[ti, di].T
            wvel = wvel_data[ti, di].T
            theta = theta_data[ti, di].T

            delayed_tasks[(ti, di)] =(
                dask.delayed(run_parallel_task)(
                    uvel,vvel,wvel,theta,
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
    df_catalogue_level0 = pd.concat([results[keys] for keys in results], ignore_index=True)

    # ---------------------------------
    # Prepare output folder + CSV
    output_path = os.path.join(lvl0_out_dir, f'lvl0_{start_date_str}_{end_date_str}.csv')
    df_catalogue_level0.to_csv(output_path, index=False)

    # ---------------------------------
    print(f'Checking if MITgcm diverged... ({get_str_current_time()})')
    last_theta_slice = ds.THETA.isel(time=-1, Z=0).values
    if np.all(np.isnan(last_theta_slice)):
        raise ValueError('Last time step of MITgcm results contains only NaN values.')

    # ---------------------------------
    print(f'Done. ({get_str_current_time()})')


if __name__ == "__main__":
    # multiprocessing compatibility for Windows (harmless on Linux)
    from multiprocessing import freeze_support
    freeze_support()
    main()