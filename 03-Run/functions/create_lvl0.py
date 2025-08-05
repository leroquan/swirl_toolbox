def compute_ke_snapshot(uvel, vvel, wvel, dx, dy, dz):
    ke = 0.5 * (uvel ** 2 + vvel ** 2 + wvel ** 2) * dx * dy * dz  # This gives J per cell

    return ke / 1e6  # Convert to MJ


def translate_rotation_direction(eddy_orientation: int):
    return 'clockwise' if eddy_orientation == -1 else 'anticlockwise'


# Helper function to extract eddy info into a row
def extract_eddy_data(indices_eddy, eddy, date, depth, dz, ke_grid_megajoules, surface_cell, theta, id_level0=0):
    vortex_indices = tuple(eddy.vortex_cells.astype(int))

    ke_lake = ke_grid_megajoules.sum()
    ke_eddy = ke_grid_megajoules[vortex_indices[0], vortex_indices[1]].sum()

    mean_temperature_eddy = theta[vortex_indices[0], vortex_indices[1]].mean()
    mean_temperature_lake = theta[theta > 0].mean()

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
        'kinetic_energy_eddy_[MJ]': float(ke_eddy),
        'kinetic_energy_lake_[MJ]': float(ke_lake),
        'mean_temperature_eddy_[°C]': mean_temperature_eddy,
        'mean_temperature_lake_[°C]': mean_temperature_lake,
        'i_eddy_cells': eddy.vortex_cells[0],
        'j_eddy_cells': eddy.vortex_cells[1]
    }