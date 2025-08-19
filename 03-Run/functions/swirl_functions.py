import swirl

def run_swirl(u_plot, v_plot, dx, dy, swirl_params_path):
    vortices = swirl.Identification(v=[u_plot, v_plot],
                                    grid_dx=[dx, dy],
                                    param_file=swirl_params_path,
                                    verbose=False)
    vortices.run()
    return vortices
