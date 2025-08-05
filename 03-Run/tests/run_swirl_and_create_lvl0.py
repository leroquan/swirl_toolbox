import pandas as pd
import os
import time

import xmitgcm as xm
import swirl

import dask
from dask import delayed, compute
dask.config.set(scheduler='processes', num_workers=576)


datapath = r"../run"
gridpath = r"../run"
ref_date = "2024-03-01 0:0:0"
dt_mitgcm_results = 60
endian = '>'
swirl_params_file_name = 'swirl_03'
model = 'geneva_200m'
output_folder = r'./'


def run_swirl(u_plot, v_plot, dx, dy, swirl_params_file):
    vortices = swirl.Identification(v=[u_plot, v_plot],
                                    grid_dx=[dx, dy],
                                    param_file=f'./swirl_params/{swirl_params_file}.param',
                                    verbose=False)
    vortices.run()
    return vortices


# Extract and compute numpy arrays BEFORE passing to dask (improves reading data time)
def load_input_data(ds_mitgcm, time_indices, depth_indices):
    dx = ds_mitgcm.dxC.values[0][0]
    dy = ds_mitgcm.dyC.values[0][0]
    dz = ds_mitgcm.drC.values
    uvel_data = ds_mitgcm['UVEL'].isel(time=time_indices, Z=depth_indices).fillna(0).values
    vvel_data = ds_mitgcm['VVEL'].isel(time=time_indices, Z=depth_indices).fillna(0).values
    wvel_data = ds_mitgcm['WVEL'].isel(time=time_indices, Zl=depth_indices).fillna(0).values

    return dx, dy, dz, uvel_data, vvel_data, wvel_data

def compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz):
    ke = 0.5 * (uvel ** 2 + vvel ** 2 + wvel ** 2) * dx * dy * dz  # This gives J per cell

    return ke / 1e6  # Convert to MJ

def translate_rotation_direction(eddy_orientation: int):
    return 'clockwise' if eddy_orientation == -1 else 'anticlockwise'

# Helper function to extract eddy info into a row
def extract_eddy_data(id_level0, indices_eddy, eddy, date, depth, dz, ke_grid_megajoules, surface_cell):
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

def run_swirl_and_create_lvl0(uvel, vvel, wvel, dx, dy, dz, swirl_params_file, id_level0, date, depth, t_index, d_index):
    eddies = run_swirl(uvel, vvel, dx, dy, swirl_params_file)
    if not eddies:  # empty list
        return pd.DataFrame()  # optionally: return with predefined columns

    ke_grid = compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz)
    eddy_rows = []
    for eddy_index in range(len(eddies)):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(id_level0, indices_eddy, eddies[eddy_index], date, depth, dz, ke_grid, dx*dy)
        eddy_rows.append(row_data)

    return pd.concat([pd.DataFrame([row]) for row in eddy_rows], ignore_index=True)

def main():
    #---------------------------------
    print('Opening MITgcm results...')
    ds_mitgcm = xm.open_mdsdataset(
        datapath,
        grid_dir=gridpath,
        ref_date=ref_date,
        prefix='3Dsnaps',
        delta_t=dt_mitgcm_results,
        endian=endian)


    # ---------------------------------
    print('Detecting eddies and creating level 0 catalogue...')
    depth_indices = range(len(ds_mitgcm.Z.values))
    time_indices = range(len(ds_mitgcm.time.values))

    start_opening_data = time.time()
    dx, dy, dz_array, uvel_data, vvel_data, wvel_data = load_input_data(ds_mitgcm, time_indices, depth_indices)
    print(f"Swirl opening data time: {time.time() - start_opening_data:.6f} seconds")

    tasks = {}
    for ti, t_idx in enumerate(time_indices):
        date = pd.Timestamp(ds_mitgcm.time.values[ti]).to_pydatetime()
        for di, d_idx in enumerate(depth_indices):
            depth = float(ds_mitgcm.Z.values[di])
            dz = dz_array[di]
            uvel = uvel_data[ti, di].T
            vvel = vvel_data[ti, di].T
            wvel = wvel_data[ti, di].T
            tasks[(t_idx, d_idx)] = delayed(run_swirl_and_create_lvl0)(uvel, vvel, wvel, dx, dy, dz, swirl_params_file_name, 0, date, depth, ti, di)

    # Compute all tasks in parallel
    start = time.time()
    results = compute(*tasks.values())
    print(f"Parallel execution time: {time.time() - start:.6f} seconds")

    # Create final DataFrame
    df_catalogue_level0 = pd.concat([row for row in results], ignore_index=True)

    date_str = ds_mitgcm.time.values[0].astype('datetime64[ms]').astype(object).strftime('%Y%m%d')
    lvl0_output_filename = f'{model}_{swirl_params_file_name}_day{date_str}_lvl0.csv'
    output_path = os.path.join(output_folder, lvl0_output_filename)

    # ---------------------------------
    print(f'Saving catalogue level 0 to {output_path}...')
    df_catalogue_level0.to_csv(output_path, index=False)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # optional on Linux/macOS
    main()