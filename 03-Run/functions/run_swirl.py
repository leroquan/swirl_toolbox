import swirl

def run_swirl(u_plot, v_plot, dx, dy, swirl_params_file):
    vortices = swirl.Identification(v=[u_plot, v_plot],
                                    grid_dx=[dx, dy],
                                    param_file=f'../postprocessing/swirl_params/{swirl_params_file}.param',
                                    verbose=False)
    vortices.run()
    return vortices
