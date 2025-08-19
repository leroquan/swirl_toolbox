import matplotlib.pyplot as plt
import numpy as np


def plot_map_swirl(uvel, vvel, vortices, title, stream_density):
    # rortex
    plt.close('all')
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(vortices.rortex[0].T, cmap='PiYG', vmin=-0.001, vmax=0.001)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_label('Rortex R')

    # streamlines
    xgrid, ygrid = np.meshgrid(range(uvel.shape[1]), range(uvel.shape[0]))
    plt.streamplot(xgrid, ygrid, uvel, vvel,
                   density=stream_density, color='black', linewidth=0.5,
                   arrowsize=0.7, arrowstyle='->')

    # Extent
    for vortex in vortices:
        plt.scatter(vortex.vortex_cells[0], vortex.vortex_cells[1], s=0.2, c='blue')
        plt.scatter(vortex.center[0], vortex.center[1], marker='*', s=50, c='red', zorder=10)

    plt.text(0.02, 0.98, f'Z=0m', transform=plt.gca().transAxes, ha='left', va='top')
    plt.title(title)

    return fig