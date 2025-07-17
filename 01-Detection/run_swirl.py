import sys

import swirl
import numpy as np
import pickle

sys.path.append('../')
from utils_mitgcm import open_mitgcm_ds_from_config


def run_swirl(data_snapshot):
    u_plot = data_snapshot['UVEL'].values
    v_plot = data_snapshot['VVEL'].values
    u_plot = np.where(np.isnan(u_plot), 0, u_plot).T
    v_plot = np.where(np.isnan(v_plot), 0, v_plot).T
    dx = data_snapshot.dxC.values[0][0]
    dy = data_snapshot.dyC.values[0][0]

    vortices = swirl.Identification(v=[u_plot, v_plot],
                                    grid_dx=[dx, dy],
                                    param_file=f'./swirl_params/{params_file}.param',
                                    verbose=True)
    vortices.run()

    return vortices


if __name__ == '__main__':
    model = 'geneva_200m'
    params_file = 'swirl_03'

    mitgcm_config, ds_mitgcm = open_mitgcm_ds_from_config('../config.json', model)
    depth_indices = range(len(ds_mitgcm.Z.values))
    time_indices = range(len(ds_mitgcm.time.values))

    eddies = {
        'depth_indices': depth_indices,
        'time_indices': time_indices
    }
    for t_idx in time_indices:
        eddies[t_idx] = {
            d_idx: run_swirl(ds_mitgcm.isel(time=t_idx, Z=d_idx))
            for d_idx in depth_indices
        }


    # Save the eddies dictionary to a file
    with open(rf'../Outputs/{model}_{params_file}_eddies.pkl', 'wb') as f:
        pickle.dump(eddies, f)