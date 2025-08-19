import numpy as np
import pandas as pd
import xarray as xr
import os
import json


class MitgcmGrid:
    """Class representing an MITgcm grid, with optional loading from .npy files."""

    def __init__(self):
        """Initialize an empty MITgcm grid."""
        self.x = np.array([])
        self.y = np.array([])
        self.lat_grid = np.array([])
        self.lon_grid = np.array([])
        self.dz = np.array([])
        self.parameters = {}

    def load_from_path(self, path_grid: str):
        """
        Load grid data from a given folder containing .npy files.

        Args:
            path_grid (str): Path to the folder containing the grid files.

        Raises:
            FileNotFoundError: If any required grid file is missing.
            RuntimeError: If loading fails due to other errors.
        """
        try:
            self.x = np.load(os.path.join(path_grid, 'x.npy'))
            self.y = np.load(os.path.join(path_grid, 'y.npy'))
            self.lat_grid = np.load(os.path.join(path_grid, 'lat_grid.npy'))
            self.lon_grid = np.load(os.path.join(path_grid, 'lon_grid.npy'))
            self.dz = pd.read_csv(os.path.join(path_grid, 'dz.csv'), header=None).to_numpy()
            with open(os.path.join(path_grid, 'parameters.json'), 'r') as file:
                self.parameters = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing grid file: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading grid data: {e}") from e


def get_mitgcm_grid(path_folder_grid: str) -> MitgcmGrid:
    grid = MitgcmGrid()
    grid.load_from_path(path_folder_grid)
    return grid


def reformat_and_save_mitgcm_results(uvel, vvel, wvel, theta, times, depths, grid_folder, output_path, nodata=-999.0):
    grid = get_mitgcm_grid(os.path.join(grid_folder, "grid"))

    # Shift velocities to cell centers
    if uvel.shape == vvel.shape:
        uvel_shifted = np.pad(uvel[..., 1:], ((0,0),(0,0), (0,0), (0,1)), constant_values=0)
        vvel_shifted = np.pad(vvel[..., 1:, :], ((0,0),(0,0), (0,1), (0,0)), constant_values=0)
    else:
        uvel_shifted = uvel[..., 1:]
        vvel_shifted = vvel[..., 1:, :]
    uvel = (uvel + uvel_shifted) / 2
    vvel = (vvel + vvel_shifted) / 2

    w = wvel.copy()
    t = theta.copy()

    # Optional rotation
    if "rotation" in grid.parameters:
        print(f"Rotating u,v by {-grid.parameters['rotation']}°")
        theta_rad = np.deg2rad(-grid.parameters["rotation"])
        u = uvel * np.cos(theta_rad) - vvel * np.sin(theta_rad)
        v = uvel * np.sin(theta_rad) + vvel * np.cos(theta_rad)
    else:
        u, v = uvel, vvel

    # Apply nodata mask
    mask = (t == 0.0) | np.isnan(t)
    for arr in [t, w, u, v]:
        arr[mask] = nodata

    # Build dataset
    ds = xr.Dataset(
        {
            "t": (("time", "depth", "Y", "X"), t, {"units": "°C", "long_name": "Temperature"}),
            "u": (("time", "depth", "Y", "X"), u, {"units": "m/s", "long_name": "Eastward velocity"}),
            "v": (("time", "depth", "Y", "X"), v, {"units": "m/s", "long_name": "Northward velocity"}),
            "w": (("time", "depth", "Y", "X"), w, {"units": "m/s", "long_name": "Vertical velocity"}),
        },
        coords={
            "time": ("time", times, {"long_name": "time"}),
            "depth": ("depth", depths, {"units": "m", "long_name": "Depth below surface"}),
            "lat": (("Y", "X"), grid.lat_grid, {"long_name": "Latitude"}),
            "lng": (("Y", "X"), grid.lon_grid, {"long_name": "Longitude"}),
        },
        attrs={"MITgcm_version": "MITgcm-checkpoint67z"},
    )

    # Ensure NaN instead of nodata if you prefer CF-compliant missing values
    ds = ds.where(ds != nodata)

    # Save to NetCDF
    ds.to_netcdf(output_path, format="NETCDF4", encoding={
        "time": {"units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"},
        "t": {"zlib": True, "complevel": 4, "_FillValue": nodata},
        "u": {"zlib": True, "complevel": 4, "_FillValue": nodata},
        "v": {"zlib": True, "complevel": 4, "_FillValue": nodata},
        "w": {"zlib": True, "complevel": 4, "_FillValue": nodata},
    })
