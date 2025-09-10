import xarray as xr
import numpy as np

from .open_files_functions import get_mitgcm_grid


def center_mitgcm_results(ds):
    u_shift = (ds['UVEL']
               .rolling(XG=2, center=False).mean()
               .shift(XG=-1)  # shift forward so it’s (i, i+1) instead of (i-1, i)
               .rename({"XG": "XC"})
               )
    u_shift['XC'] = ds['XC']

    v_shift = (ds['VVEL']
               .rolling(YG=2, center=False).mean()
               .shift(YG=-1)  # shift forward so it’s (i, i+1) instead of (i-1, i)
               .rename({"YG": "YC"}))
    v_shift['YC'] = ds['YC']

    w_shift = (ds['WVEL']
               .shift(Zl=1).fillna(0)  # shift to get a 0 at the surface
               .rolling(Zl=2, center=False).mean()
               .shift(Zl=-1)  # shift back
               .rename({"Zl": "Z"}))
    w_shift['Z'] = ds['Z']

    return u_shift, v_shift, w_shift


def rotate_mitgcm_results(angle_in_degrees, u, v):
    print(f"Rotating u,v by {angle_in_degrees}°")
    theta_rad = np.deg2rad(angle_in_degrees)
    u_rot = u * np.cos(theta_rad) - v * np.sin(theta_rad)
    v_rot = u * np.sin(theta_rad) + v * np.cos(theta_rad)

    return u_rot, v_rot


def reformat_mitgcm_results(ds, times, depths, grid_folder, nodata=-999.0):
    grid = get_mitgcm_grid(grid_folder)

    u_shift, v_shift, w_shift = center_mitgcm_results(ds)

    # Optional rotation
    if "rotation" in grid.parameters:
        angle_in_degrees = -grid.parameters['rotation']
        u, v = rotate_mitgcm_results(angle_in_degrees, u_shift, v_shift)
        w = w_shift
    else:
        u, v, w = u_shift, v_shift, w_shift

    # Apply nodata mask
    print(f"Applying nodata mask")
    theta = ds['THETA']
    mask = (theta == 0.0) | np.isnan(theta)
    for arr in [theta, w, u, v]:
        arr = arr.where(~mask, nodata)

    # Build dataset
    print(f"Building dataset")
    theta.attrs = {"units": "°C", "long_name": "Temperature"}
    u.attrs = {"units": "m/s", "long_name": "Eastward velocity"}
    v.attrs = {"units": "m/s", "long_name": "Northward velocity"}
    w.attrs = {"units": "m/s", "long_name": "Vertical velocity"}

    # Build dataset
    ds = xr.Dataset(
        {
            "t": (("time", "depth", "Y", "X"), theta.data, {"units": "°C", "long_name": "Temperature"}),
            "u": (("time", "depth", "Y", "X"), u.data, {"units": "m/s", "long_name": "Eastward velocity"}),
            "v": (("time", "depth", "Y", "X"), v.data, {"units": "m/s", "long_name": "Northward velocity"}),
            "w": (("time", "depth", "Y", "X"), w.data, {"units": "m/s", "long_name": "Vertical velocity"}),
        },
        coords={
            "time": ("time", times, {"long_name": "time"}),
            "depth": ("depth", depths, {"units": "m", "long_name": "Depth below surface"}),
            "lat": (("Y", "X"), grid.lat_grid, {"long_name": "Latitude"}),
            "lng": (("Y", "X"), grid.lon_grid, {"long_name": "Longitude"}),
        },
        attrs={"MITgcm_version": "MITgcm-checkpoint67z"},
    )

    return ds