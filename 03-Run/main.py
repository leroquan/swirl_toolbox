"""
Created on 30.07.2025 by Anne Leroquais (EAWAG)
Here, we use netcdf results from MITgcm to detect eddies in lakes.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

import json

from functions import run_swirl
from functions  import plot_map_swirl
from functions import load_input_data_netcdf, load_input_data_binary
from functions import reformat_and_save_mitgcm_results
from functions import compute_ke_snapshot, extract_eddy_data

import dask
from dask import delayed, compute


O_MITGCM_FOLDER_NAME = 'mitgcm_results'
O_GRID_FOLDER_NAME = 'grid'
O_LVL0_FOLDER_NAME = 'eddy_catalogues_lvl0'



def run_parallel_task(uvel, vvel, wvel, theta,
                      dx, dy, dz,
                      swirl_params_path,
                      date, depth,
                      t_index, d_index,
                      output_folder,
                      id_level0=0):
    eddies = run_swirl(uvel, vvel, dx, dy, swirl_params_path)
    if not eddies:  # empty list
        return pd.DataFrame()  # optionally: return with predefined columns

    ke_grid = compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz)
    eddy_rows = []
    for eddy_index in range(len(eddies)):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(indices_eddy, eddies[eddy_index], date, depth, dz, ke_grid, dx*dy, theta, id_level0)
        eddy_rows.append(row_data)

    if d_index == 0:
        fig = plot_map_swirl(uvel.T, vvel.T, eddies, date, 6)
        fig.savefig(os.path.join(output_folder, f'figures/map_swirl_{date}.png'))

    return pd.concat([pd.DataFrame([row]) for row in eddy_rows], ignore_index=True)


def get_str_current_time():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def main(config_path = '..//postprocessing//config_postprocessing.json'):
    with open(config_path, 'r') as file:
        pp_config = json.load(file)

    dask.config.set(scheduler='threads')
    #---------------------------------
    # Load data
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
        raise ValueError(f'"mitgcm_output_format" must be either "netcdf" or "binary" : {pp_config["mitgcm_output_format"]}')

    start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime('%Y%m%d')
    end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime('%Y%m%d')

    # ---------------------------------
    # Save mitgcm results
    if pp_config['save_nc_mitgcm'] == "True":
        print(f'Saving merged mitgcm results... ({get_str_current_time()})')
        reformat_and_save_mitgcm_results(swirl_input_data.uvel_data, swirl_input_data.vvel_data, swirl_input_data.wvel_data,
                                         swirl_input_data.theta_data,
                                         swirl_input_data.times, swirl_input_data.depths,
                                         pp_config['grid_folder'],
                                         os.path.join(pp_config['output_folder'],
                                                      O_MITGCM_FOLDER_NAME,
                                                      rf"mitgcm_{start_date_str}_{end_date_str}.nc"),
                                         nodata=-999.0)

    # ---------------------------------
    # Prepare parallel tasks
    print(f'Preparing parallel tasks... ({get_str_current_time()})')
    nb_cores = pp_config['px'] * pp_config['py']
    print(f'...using {nb_cores} cores...')
    dask.config.set(scheduler='processes', num_workers=nb_cores)

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
            tasks[(t_idx, d_idx)] = delayed(run_parallel_task)(uvel, vvel, wvel, theta,
                                                               swirl_input_data.dx, swirl_input_data.dy, dz,
                                                               pp_config['swirl_params_path'],
                                                               date, depth,
                                                               ti, di,
                                                               pp_config['output_folder'])

    # ---------------------------------
    # Compute all tasks in parallel
    print(f'Computing parallel tasks... ({get_str_current_time()})')
    results = compute(*tasks.values())

    # ---------------------------------
    # Save catalogue

    # Create final DataFrame
    df_catalogue_level0 = pd.concat([row for row in results], ignore_index=True)

    output_path = os.path.join(pp_config['output_folder'], O_LVL0_FOLDER_NAME, f'lvl0_{start_date_str}_{end_date_str}.csv')
    print(f'Saving catalogue level 0 to {output_path}... ({get_str_current_time()})')
    df_catalogue_level0.to_csv(output_path, index=False)

    # ---------------------------------
    # Check if MITgcm diverged
    if np.all(np.isnan(swirl_input_data.theta_data[swirl_input_data.time_indices[-1], swirl_input_data.depth_indices[0]])):
        raise ValueError('Last time step of MITgcm results contains only Nan values.')
    
    print(f'Done. ({get_str_current_time()})')


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # optional on Linux/macOS
    main()