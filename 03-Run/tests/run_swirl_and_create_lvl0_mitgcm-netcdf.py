"""
Created on 30.07.2025 by Anne Leroquais (EAWAG)
Here, we use netcdf results from MITgcm to detect eddies in lakes.
"""

import pandas as pd
import os
import time
from collections import namedtuple

import xarray as xr
import swirl

import dask
from dask import delayed, compute
dask.config.set(scheduler='processes', num_workers=576)


mitgcm_nc_results_path = r"../run"
swirl_params_file_name = 'swirl_03'
model = 'geneva_200m'
output_folder = r'./'

SwirlInputData = namedtuple('SwirlInputData', [
    'times', 'depths', 'time_indices', 'depth_indices',
    'dx', 'dy', 'dz_array',
    'uvel_data', 'vvel_data', 'wvel_data'
])


def run_swirl(u_plot, v_plot, dx, dy, swirl_params_file):
    vortices = swirl.Identification(v=[u_plot, v_plot],
                                    grid_dx=[dx, dy],
                                    param_file=f'./swirl_params/{swirl_params_file}.param',
                                    verbose=False)
    vortices.run()
    return vortices


# Extract and compute numpy arrays BEFORE passing to dask (improves reading data time)
def load_input_data(mitgcm_nc_results_path, time_indices=None, depth_indices=None):
    ds_mitgcm = xr.open_mfdataset(os.path.join(mitgcm_nc_results_path ,'3Dsnaps*.nc'))
    ds_grid = xr.open_dataset(os.path.join(mitgcm_nc_results_path, 'grid_merged.nc'))

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

    return SwirlInputData(times, depths,
                          time_indices,depth_indices,
                          dx, dy, dz_array,
                          uvel_data, vvel_data, wvel_data)


def compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz):
    ke = 0.5 * (uvel ** 2 + vvel ** 2 + wvel ** 2) * dx * dy * dz  # This gives J per cell

    return ke / 1e6  # Convert to MJ


def translate_rotation_direction(eddy_orientation: int):
    return 'clockwise' if eddy_orientation == -1 else 'anticlockwise'


# Helper function to extract eddy info into a row
def extract_eddy_data(indices_eddy, eddy, date, depth, dz, ke_grid_megajoules, surface_cell, id_level0=0):
    vortex_indices = tuple(eddy.vortex_cells.astype(int))
    ke_eddy = ke_grid_megajoules[vortex_indices[0], vortex_indices[1]].sum()
    surface_area = len(eddy.vortex_cells[0]) * surface_cell

    return {
        'id': id_level0,
        'time_index': indices_eddy[0],
        'depth_index': indices_eddy[1],
        'eddy_index': indices_eddy[2],
        'date': date,
        'depth_[m]': depth,
        'xc': eddy.center[0],
        'yc': eddy.center[1],
        'surface_area_[m2]': float(surface_area),
        'volume_slice_[m3]': float(surface_area * dz),
        'rotation_direction': translate_rotation_direction(eddy.orientation),
        'kinetic_energy_[MJ]': float(ke_eddy),
        'i_eddy_cells': eddy.vortex_cells[0],
        'j_eddy_cells': eddy.vortex_cells[1]
    }


def run_swirl_and_create_lvl0(uvel, vvel, wvel,
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
        row_data = extract_eddy_data(indices_eddy, eddies[eddy_index], date, depth, dz, ke_grid, dx*dy, id_level0)
        eddy_rows.append(row_data)

    return pd.concat([pd.DataFrame([row]) for row in eddy_rows], ignore_index=True)


def main():
    #---------------------------------
    print('Loading input data...')
    start_opening_data = time.time()
    swirl_input_data = load_input_data(mitgcm_nc_results_path)
    print(f"Swirl opening data time: {time.time() - start_opening_data:.6f} seconds")

    # ---------------------------------
    print('Detecting eddies and creating level 0 catalogue...')
    tasks = {}
    for ti, t_idx in enumerate(swirl_input_data.time_indices):
        date = pd.Timestamp(swirl_input_data.times[ti]).to_pydatetime()
        for di, d_idx in enumerate(swirl_input_data.depth_indices):
            depth = float(swirl_input_data.depths[di])
            dz = swirl_input_data.dz_array[di]
            uvel = swirl_input_data.uvel_data[ti, di].T
            vvel = swirl_input_data.vvel_data[ti, di].T
            wvel = swirl_input_data.wvel_data[ti, di].T
            tasks[(t_idx, d_idx)] = delayed(run_swirl_and_create_lvl0)(uvel, vvel, wvel,
                                                                       swirl_input_data.dx, swirl_input_data.dy, dz,
                                                                       swirl_params_file_name,
                                                                       date, depth,
                                                                       ti, di)

    # Compute all tasks in parallel
    start = time.time()
    results = compute(*tasks.values())
    print(f"Parallel execution time: {time.time() - start:.6f} seconds")

    # Create final DataFrame
    df_catalogue_level0 = pd.concat([row for row in results], ignore_index=True)

    date_str = swirl_input_data.times[0].astype('datetime64[ms]').astype(object).strftime('%Y%m%d')
    lvl0_output_filename = f'{model}_{swirl_params_file_name}_day{date_str}_lvl0.csv'
    output_path = os.path.join(output_folder, lvl0_output_filename)

    # ---------------------------------
    print(f'Saving catalogue level 0 to {output_path}...')
    df_catalogue_level0.to_csv(output_path, index=False)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # optional on Linux/macOS
    main()