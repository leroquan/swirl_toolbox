"""
Created on 30.07.2025 by Anne Leroquais (EAWAG) - memory-safe dask version
Modified on 08.10.2025 by Flavio Calvo (UNIL), with great help from Anthropic/Sonnet 4.5 (code generation) and black (code formatting)

Notes:
- Expects load_input_data_netcdf / load_input_data_binary to return lazy xarray DataArrays
  chunked so that .isel(time=..., Z=...) returns a small Dask graph (1 plane).
"""

import os
import sys
import time
from datetime import datetime
import ctypes
import ctypes.util
import gc

import json
import psutil
import traceback

import pandas as pd
import numpy as np

import dask
from dask.distributed import Client, LocalCluster, as_completed

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, better for saving files
import matplotlib.pyplot as plt

from functions import run_swirl, plot_map_swirl
from functions import load_input_data_netcdf, load_input_data_binary
from functions import reformat_mitgcm_results
from functions import compute_ke_snapshot, extract_eddy_data
from functions import save_to_netcdf

O_MITGCM_FOLDER_NAME = "mitgcm_results"
O_GRID_FOLDER_NAME = "grid"
O_LVL0_FOLDER_NAME = "eddy_catalogues_lvl0"


def can_use_malloc_trim():
    """Check if malloc_trim is available (cached at module level)"""
    libc_name = ctypes.util.find_library("c")
    return libc_name is not None


# Use malloc_trim to aggressively return freed memory (with the GNU allocator only)
HAS_MALLOC_TRIM = False  # can_use_malloc_trim()
print(f"malloc_trim available: {HAS_MALLOC_TRIM}")


def get_str_current_time():
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")


def run_parallel_task(
    u,
    v,
    w,
    theta,
    dx,
    dy,
    dz,
    swirl_params_path,
    date,
    depth,
    t_index,
    d_index,
    output_folder,
    max_n_evc_points=30000,
    overhead_factor=1.25,
    id_level0=0,
    verbose=False,
):
    import uuid
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Eddy detection & feature extraction (same as before)
    if verbose:
        print(f"Run swirl for t={t_index}, d={d_index}")
        print(f"[TASK] About to call run_swirl...", flush=True)
    eddies = run_swirl(
        u,
        v,
        dx,
        dy,
        swirl_params_path,
        max_n_evc_points,
        overhead_factor,
        t_index,
        d_index,
        str(uuid.uuid4())[:8],
    )
    if verbose:
        print(f"[TASK] run_swirl RETURNED!", flush=True)
    # Check using attribute, not boolean
    if hasattr(eddies, "n_vortices") and eddies.n_vortices == 0:
        print(f"[TASK] No vortices found, returning empty", flush=True)
        return pd.DataFrame()

    if verbose:
        print(f"[TASK] Processing {eddies.n_vortices} vortices...", flush=True)
    if not eddies:
        return pd.DataFrame()  # empty result for this slice

    # print(f'Compute KE for t={t_index}, d={d_index}')
    ke_grid = compute_ke_snapshot(u, v, w, dx, dy, dz)

    # print(f'Extract eddy data for t={t_index}, d={d_index}')
    eddy_rows = []
    for eddy_index, eddy in enumerate(eddies):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(
            indices_eddy, eddy, date, depth, dz, ke_grid, dx * dy, theta, id_level0
        )
        eddy_rows.append(row_data)

    # Optional map output for first depth
    if d_index == -1:
        if verbose:
            print(f"Map t={t_index}, d={d_index}")
        fig = plot_map_swirl(u.T, v.T, eddies, date, 6)
        out_fig_dir = os.path.join(output_folder, "figures")
        os.makedirs(out_fig_dir, exist_ok=True)
        fig_path = os.path.join(out_fig_dir, f"map_swirl_{date}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # IMPORTANT: free matplotlib memory

        # More aggressive cleanup
        plt.close("all")  # Close all figures
        gc.collect()  # Force garbage collection

    del u, v, w, theta, ke_grid
    gc.collect()  # Force garbage collection

    # print(f'Concatenating t={t_index}, d={d_index}')
    # return pd.concat([pd.DataFrame([r]) for r in eddy_rows], ignore_index=True)
    return pd.DataFrame(eddy_rows)  # Create DataFrame directly from list of dicts


def get_number_of_cores(pp_config):
    """
    Get number of cores from SLURM environment or config file.

    Parameters:
    -----------
    pp_config : dict
        Configuration dictionary

    Returns:
    --------
    int : Number of cores to use
    """
    verbose = pp_config.get("verbose", False)

    # Try SLURM environment variables (in order of preference)
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")

    if slurm_cpus_per_task:
        # Most common for single-node parallel jobs
        nb_cores = int(slurm_cpus_per_task)
        if verbose:
            print(f"Using SLURM_CPUS_PER_TASK: {nb_cores} cores")
        return nb_cores

    elif slurm_ntasks:
        # For multi-task jobs (MPI-style)
        nb_cores = int(slurm_ntasks)
        if verbose:
            print(f"Using SLURM_NTASKS: {nb_cores} cores")
        return nb_cores

    elif slurm_cpus_on_node:
        # Total CPUs on the node
        nb_cores = int(slurm_cpus_on_node)
        if verbose:
            print(f"Using SLURM_CPUS_ON_NODE: {nb_cores} cores")
        return nb_cores

    else:
        # Fall back to config file
        nb_cores = int(pp_config.get("px", 1)) * int(pp_config.get("py", 1))
        if verbose:
            print(
                f"No SLURM detected. Using config file: px={pp_config.get('px', 1)} × py={pp_config.get('py', 1)} = {nb_cores} cores"
            )
        return nb_cores


def get_available_memory(verbose=False):
    """
    Get available memory respecting SLURM allocation.

    Returns:
    --------
    float : Available memory in bytes
    """
    # Check for SLURM environment variables
    slurm_mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    slurm_mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_ntasks = os.environ.get("SLURM_NTASKS")

    if slurm_mem_per_node:
        # Memory per node in MB
        memory_bytes = int(slurm_mem_per_node) * 1024 * 1024
        if verbose:
            print(f"Detected SLURM allocation: {slurm_mem_per_node} MB per node")
        return memory_bytes

    elif slurm_mem_per_cpu and slurm_cpus_per_task:
        # Memory per CPU × number of CPUs
        mem_per_cpu_mb = int(slurm_mem_per_cpu)
        cpus = int(slurm_cpus_per_task)
        memory_bytes = mem_per_cpu_mb * cpus * 1024 * 1024
        if verbose:
            print(
                f"Detected SLURM allocation: {mem_per_cpu_mb} MB/CPU × {cpus} CPUs = {memory_bytes / 1e9:.1f} GB"
            )
        return memory_bytes

    elif slurm_mem_per_cpu and slurm_ntasks:
        # Memory per CPU × number of tasks
        mem_per_cpu_mb = int(slurm_mem_per_cpu)
        ntasks = int(slurm_ntasks)
        memory_bytes = mem_per_cpu_mb * ntasks * 1024 * 1024
        if verbose:
            print(
                f"Detected SLURM allocation: {mem_per_cpu_mb} MB/CPU × {ntasks} tasks = {memory_bytes / 1e9:.1f} GB"
            )
        return memory_bytes

    else:
        # Fall back to psutil (local machine or no SLURM)
        memory_bytes = psutil.virtual_memory().available
        if verbose:
            print(
                f"No SLURM memory allocation detected. Using available system memory: {memory_bytes / 1e9:.1f} GB"
            )
        return memory_bytes


def estimate_batch_size(
    ds,
    n_times,
    n_depths,
    nb_cores=1,
    safety_factor=0.3,
    account_for_overhead=False,
    target_tasks_multiplier=3,
    max_n_evc_points=None,
    overhead_factor=None,
    verbose=False,
):
    """
    Estimate batch size based on both memory constraints AND parallelism needs.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset
    n_times : int
        Number of time steps
    n_depths : int
        Number of depth levels
    nb_cores : int
        Number of workers (1 = main process, >1 = distributed)
    safety_factor : float
        Fraction of memory to use (0.2-0.3 for workers, 0.7-0.8 for main)
    account_for_overhead : bool
        If True, explicitly subtract Python overhead (~400MB) before calculating
        Use True for worker processes, False for main process
    target_tasks_multiplier : int
        Target number of tasks = nb_cores × this multiplier
        (e.g., 3 means aim for 3× as many tasks as workers for load balancing)
    max_n_evc_points : int
        Maximum number of EVC points for SWIRL clustering
        Used to estimate worst-case SWIRL memory consumption
    overhead_factor : float
        Multiplicative factor for SWIRL clustering memory overhead
        Accounts for intermediate allocations and GC behavior
    """

    # Get memory estimation
    sample_uvel = ds.UVEL.isel(time=0, Z=0)
    sample_vvel = ds.VVEL.isel(time=0, Z=0)
    sample_wvel = ds.WVEL.isel(time=0, Zl=0)
    sample_theta = ds.THETA.isel(time=0, Z=0)

    # Calculate memory per timestep based on actual array sizes and dtypes
    # For one time step, all depths
    memory_per_timestep = (
        sample_uvel.size * n_depths * sample_uvel.dtype.itemsize
        + sample_vvel.size * n_depths * sample_vvel.dtype.itemsize
        + sample_wvel.size * n_depths * sample_wvel.dtype.itemsize
        + sample_theta.size * n_depths * sample_theta.dtype.itemsize
    )

    if max_n_evc_points is not None and overhead_factor is not None:

        # Estimate SWIRL clustering memory per timestep
        # Formula: overhead_factor × 2.0 × (N² × 8 bytes)
        # where N = max_n_evc_points (worst case per pixel)
        swirl_memory_per_pixel = overhead_factor * 2.0 * (max_n_evc_points**2) * 8

        # Total SWIRL memory for one timestep (all depths)
        swirl_memory_per_timestep = swirl_memory_per_pixel * n_depths

        # Total memory per timestep including input data + SWIRL processing
        total_memory_per_timestep = memory_per_timestep + swirl_memory_per_timestep

        if verbose:
            print(
                f"  Data types: UVEL={sample_uvel.dtype}, VVEL={sample_vvel.dtype}, WVEL={sample_wvel.dtype}, THETA={sample_theta.dtype}"
            )

            # Print memory breakdown
            print(f"  Memory per timestep breakdown:")
            print(f"    Input data:        {memory_per_timestep / 1e9:.2f} GB")
            print(f"    SWIRL (estimated): {swirl_memory_per_timestep / 1e9:.2f} GB")
            print(
                f"      (max {max_n_evc_points} EVC points/pixel × {n_depths} depths × {overhead_factor}× overhead)"
            )
            print(f"    Total per timestep: {total_memory_per_timestep / 1e9:.2f} GB")

        # Peak memory = (batch_size × input_data) + (swirl_buffer for 1-2 pixels)
        # Safety: assume 2 pixels' worth of SWIRL can coexist if GC lags
        swirl_safety_buffer = 1.5 * swirl_memory_per_pixel

    else:
        swirl_memory_per_pixel = 0
        swirl_safety_buffer = 0
        total_memory_per_timestep = memory_per_timestep

        if verbose:
            print(
                f"  Data types: UVEL={sample_uvel.dtype}, VVEL={sample_vvel.dtype}, WVEL={sample_wvel.dtype}, THETA={sample_theta.dtype}"
            )
            print(
                f"  Memory per timestep (input data): {memory_per_timestep / 1e9:.2f} GB"
            )
            print(
                f"  SWIRL memory estimation: DISABLED (no max_n_evc_points/overhead_factor)"
            )

    # Total memory
    total_memory = get_available_memory(verbose=verbose)

    if nb_cores > 1:
        # === CONSTRAINT 1: MEMORY ===
        # Calculate maximum chunk size that fits in worker memory
        worker_memory = total_memory / nb_cores

        if account_for_overhead:
            # Explicitly subtract Python overhead for worker processes
            python_overhead = 400 * 1024 * 1024  # 400 MB
            usable_worker_memory = max(0, worker_memory - python_overhead)
            if verbose:
                print(f"  {nb_cores} workers, {worker_memory / 1e9:.2f} GB per worker")
                print(
                    f"  After Python overhead (~400 MB): {usable_worker_memory / 1e9:.2f} GB usable"
                )
        else:
            usable_worker_memory = worker_memory
            if verbose:
                print(f"  {nb_cores} workers, {worker_memory / 1e9:.2f} GB per worker")

        available_memory = worker_memory * safety_factor

        if available_memory > swirl_safety_buffer:
            memory_limited_batch = int(
                (available_memory - swirl_safety_buffer) / memory_per_timestep
            )
            memory_limited_batch = max(1, memory_limited_batch)
        else:
            # Not enough memory even for SWIRL safety buffer
            if verbose:
                print(
                    f"  ⚠️  WARNING: Available memory ({available_memory / 1e9:.2f} GB) barely covers SWIRL buffer ({swirl_safety_buffer / 1e9:.2f} GB)"
                )
            memory_limited_batch = 1

        # === CONSTRAINT 2: PARALLELISM ===
        # Calculate chunk size needed to create enough tasks for all workers
        # We want at least (nb_cores × target_tasks_multiplier) tasks
        target_num_tasks = int(nb_cores * target_tasks_multiplier)

        # If we chunk only on time (depth=-1), then n_tasks = ceil(n_times / time_chunk_size)
        # We want: n_tasks >= target_num_tasks
        # So: time_chunk_size <= n_times / target_num_tasks
        parallelism_limited_batch = max(1, n_times // target_num_tasks)

        # === CHOOSE THE MORE RESTRICTIVE CONSTRAINT ===
        batch_size = min(memory_limited_batch, parallelism_limited_batch)

        # Calculate resulting number of tasks (assuming depth_chunk_size = -1)
        resulting_num_tasks = (
            n_times + batch_size - 1
        ) // batch_size  # Ceiling division

        if verbose:
            print(f"  Memory constraint: max {memory_limited_batch} time steps/task")
            print(
                f"  Parallelism constraint: max {parallelism_limited_batch} time steps/task (target: {target_num_tasks} tasks)"
            )
            print(f"  → Choosing: {batch_size} time steps/task")
            print(
                f"  → Will create ~{resulting_num_tasks} tasks (with {n_times} time steps total)"
            )

            if resulting_num_tasks < nb_cores:
                print(
                    f"     WARNING: Only {resulting_num_tasks} tasks for {nb_cores} workers!"
                )
                print(
                    f"     Some workers will be idle. Consider reducing chunk size manually."
                )
    else:
        # Main process: only memory constraint matters (no parallelism)
        available_memory = total_memory * safety_factor

        if available_memory > swirl_safety_buffer:
            batch_size = int(
                (available_memory - swirl_safety_buffer) / memory_per_timestep
            )
            batch_size = max(1, min(batch_size, 100))
        else:
            batch_size = 1

        if verbose:
            print(f"  Main process, {total_memory / 1e9:.2f} GB available")

    if verbose:
        print(f"Estimated batch size: {batch_size} time steps")
        print(f"  Total memory per time step: {total_memory_per_timestep / 1e9:.2f} GB")

    return batch_size


def setup_dask_client_slurm(nb_cores, memory_per_worker="auto", verbose=False):
    """
    Setup Dask client with SLURM-aware memory limits.

    Parameters:
    -----------
    nb_cores : int
        Number of workers (typically from SLURM_CPUS_PER_TASK or px*py)
    memory_per_worker : str or int
        Memory limit per worker. If 'auto', divides SLURM allocation by nb_cores
    """

    if memory_per_worker == "auto":
        # Get total SLURM allocation
        total_memory = get_available_memory(verbose=verbose)

        # Divide by number of workers, with headroom
        memory_per_worker_bytes = int(
            (total_memory / nb_cores) * 0.9
        )  # 90% to leave headroom
        memory_per_worker = f"{memory_per_worker_bytes}B"

        print(
            f"Auto-calculated memory per worker: {memory_per_worker_bytes / 1e9:.2f} GB"
        )

    cluster = LocalCluster(
        n_workers=nb_cores,
        threads_per_worker=1,
        memory_limit=memory_per_worker,
        processes=True,
    )
    client = Client(cluster)

    if verbose:
        print(f"Dask cluster configured:")
        print(f"  Workers: {nb_cores}")
        print(f"  Memory per worker: {memory_per_worker}")
        print(f"  Dashboard: {client.dashboard_link}")

    return client, cluster


def create_chunked_tasks(
    swirl_input_data,
    pp_config,
    ds,
    nb_cores,
    time_chunk_size=2,
    depth_chunk_size=50,
    target_tasks_multiplier=3,
    max_n_evc_points=30000,
    overhead_factor=1.25,
    verbose=False,
):
    """
    Create chunked tasks to reduce overhead.

    Chunking Logic:
    ===============

    IF time_chunk_size == 'auto':
        1. Estimate time_chunk_size based on memory constraints and parallelism needs
        2. Handle depth_chunk_size:
           - IF depth_chunk_size == 'auto':
               - IF time chunking creates enough tasks (>= nb_cores × 3):
                   → Set depth_chunk_size = -1 (full depth, no depth chunking)
               - ELSE (time chunking insufficient):
                   → Calculate depth_chunk_size to reach target number of tasks
           - ELIF depth_chunk_size == -1:
               → Keep depth_chunk_size = -1 (full depth)
           - ELSE (depth_chunk_size is a positive integer):
               → Keep user-specified depth_chunk_size as-is

    ELSE (time_chunk_size is not 'auto', i.e., user-specified or -1):
        1. Keep time_chunk_size as-is
        2. Handle depth_chunk_size:
           - IF depth_chunk_size == 'auto' OR depth_chunk_size == -1:
               → Set depth_chunk_size = -1 (full depth, no depth chunking)
           - ELSE (depth_chunk_size is a positive integer):
               → Keep user-specified depth_chunk_size as-is

    Special Values:
    ---------------
    - 'auto': Automatically determine chunk size
    - -1: Use full dimension (no chunking on that dimension)
    - positive integer: Use that specific chunk size

    Parameters:
    -----------
    swirl_input_data : SwirlInputData
        Input data object containing times, depths, etc.
    pp_config : dict
        Configuration dictionary
    ds : xarray.Dataset
        The dataset to chunk
    nb_cores : int
        Number of workers
    time_chunk_size : int or str
        Number of time steps per task (default: 1)
    depth_chunk_size : int or str
        Number of depth levels per task (default: 1)
    target_tasks_multiplier : float
        Target number of tasks = nb_cores × this multiplier (default: 3)
        Use lower values (1.5) for fast tasks, higher (3-5) for slow tasks
    max_n_evc_points : int
        Maximum number of EVC points for SWIRL clustering
    overhead_factor : float
        Overhead factor for SWIRL memory estimation

    Returns:
    --------
    dict : Dictionary of delayed tasks
    """

    n_times = len(swirl_input_data.times)
    n_depths = len(swirl_input_data.depths)

    # Store whether values are auto
    time_is_auto = time_chunk_size == "auto"
    depth_is_auto = depth_chunk_size == "auto"

    # === HANDLE TIME CHUNKING ===
    if time_is_auto:
        if verbose:
            print(
                "  time_chunk_size='auto' → estimating based on memory AND parallelism..."
            )
        time_chunk_size = estimate_batch_size(
            ds,
            n_depths,
            n_times,
            nb_cores=nb_cores,
            safety_factor=0.3,
            account_for_overhead=True,
            target_tasks_multiplier=target_tasks_multiplier,
            max_n_evc_points=30000,
            overhead_factor=1.25,
        )

    # === HANDLE DEPTH CHUNKING ===
    if not time_is_auto:
        # Time is manual
        if depth_is_auto or depth_chunk_size == -1:
            # Depth is auto or -1: set to full depth
            if verbose:
                print(
                    f"  time_chunk_size is manual → setting depth_chunk_size=-1 (full depth)"
                )
            depth_chunk_size = -1
        else:
            # Depth is a positive number: keep it
            if verbose:
                print(
                    f"  time_chunk_size is manual → keeping depth_chunk_size={depth_chunk_size}"
                )

    elif depth_is_auto:
        # Time is auto, depth is auto: check if time chunking alone is sufficient
        # Handle -1 for time first
        effective_time_chunk = n_times if time_chunk_size == -1 else time_chunk_size

        # Calculate how many tasks we'd get with depth=-1 (full depth)
        num_time_chunks = (n_times + effective_time_chunk - 1) // effective_time_chunk

        # Only worry about parallelism if we have multiple workers
        if nb_cores == 1:
            # Single worker: no need for extra tasks
            if verbose:
                print(
                    f"  depth_chunk_size='auto' → single worker, using full depth dimension"
                )
            depth_chunk_size = -1
        else:
            # Multiple workers: check if we have enough tasks for load balancing
            target_num_tasks = int(nb_cores * target_tasks_multiplier)

            if num_time_chunks >= target_num_tasks:
                # Time chunking alone is sufficient
                if verbose:
                    print(
                        f"  depth_chunk_size='auto' → time chunking creates {num_time_chunks} tasks"
                    )
                    print(
                        f"  → Sufficient for {nb_cores} workers, using full depth dimension"
                    )
                depth_chunk_size = -1
            else:
                # Need to also chunk on depth to reach target
                depth_chunks_needed = (
                    target_num_tasks + num_time_chunks - 1
                ) // num_time_chunks
                depth_chunk_size = max(1, n_depths // depth_chunks_needed)

                total_tasks = num_time_chunks * depth_chunks_needed

                if verbose:
                    print(
                        f"  depth_chunk_size='auto' → time chunking only creates {num_time_chunks} tasks"
                    )
                    print(
                        f"  → Insufficient for {nb_cores} workers (target: {target_num_tasks} tasks)"
                    )
                    print(f"  → Also chunking depth: {depth_chunk_size} depths/task")
                    print(
                        f"  → Will create {num_time_chunks} time × {depth_chunks_needed} depth = ~{total_tasks} total tasks"
                    )

    else:
        # Time is auto, depth is a specific number: keep depth as-is
        if verbose:
            print(
                f"  depth_chunk_size={depth_chunk_size} → keeping manual depth chunking"
            )

    # === HANDLE -1 CONVENTION ===
    if time_chunk_size == -1:
        time_chunk_size = n_times
        if verbose:
            print(f"  time_chunk_size=-1 → using full time dimension: {n_times}")

    if depth_chunk_size == -1:
        depth_chunk_size = n_depths
        if verbose:
            print(f"  depth_chunk_size=-1 → using full depth dimension: {n_depths}")

    # === CREATE CHUNKS ===
    # Create time chunks
    time_chunks = []
    for t_start in range(0, n_times, time_chunk_size):
        t_end = min(t_start + time_chunk_size, n_times)
        time_chunks.append((t_start, t_end))

    # Create depth chunks
    depth_chunks = []
    for d_start in range(0, n_depths, depth_chunk_size):
        d_end = min(d_start + depth_chunk_size, n_depths)
        depth_chunks.append((d_start, d_end))

    pixels_per_task = time_chunk_size * depth_chunk_size
    total_pixels = n_times * n_depths
    total_tasks = len(time_chunks) * len(depth_chunks)

    if verbose:
        print(f"\nTask chunking summary:")
        print(
            f"  {len(time_chunks)} time chunks × {len(depth_chunks)} depth chunks = {total_tasks} total tasks"
        )
        print(
            f"  Each task processes ~{pixels_per_task} pixels ({total_pixels} pixels total)"
        )
        print(
            f"  {nb_cores} workers available → ~{total_tasks / nb_cores:.1f} tasks per worker\n"
        )

    task_dict = {}
    task_id = 0

    for t_start, t_end in time_chunks:
        for d_start, d_end in depth_chunks:

            @dask.delayed
            def process_chunk(
                t_start=t_start,
                t_end=t_end,
                d_start=d_start,
                d_end=d_end,
                max_n_evc_points=max_n_evc_points,
                overhead_factor=overhead_factor,
                verbose=verbose,
            ):
                """Process a chunk of (time, depth) pairs"""
                import os
                import uuid

                task_id = str(uuid.uuid4())[:8]  # Short unique ID
                worker_id = os.getpid()

                process = psutil.Process(os.getpid())

                # Force aggressive cleanup BEFORE starting
                gc.collect()
                gc.collect()
                gc.collect()

                # Check what's in memory
                all_objects = gc.get_objects()
                large_objects = []
                for obj in all_objects:
                    try:
                        size = sys.getsizeof(obj)
                        if size > 10 * 1024 * 1024:  # > 10 MB
                            large_objects.append((type(obj).__name__, size / 1024**2))
                    except:
                        pass

                large_objects.sort(key=lambda x: x[1], reverse=True)

                if verbose:
                    print(
                        f"[Task {task_id}] BEFORE loading - Large objects in memory:",
                        flush=True,
                    )
                    for name, size_mb in large_objects[:15]:
                        print(f"  {name}: {size_mb:.1f} MB", flush=True)

                # Before loading any data
                mem_before = process.memory_info().rss / 1024**3
                if verbose:
                    print(
                        f"[Task {task_id}] Worker {worker_id} memory BEFORE loading: {mem_before:.2f} GB"
                    )

                # Time data loading - load ENTIRE CHUNK at once
                load_start = time.time()
                uvel_chunk = ds.UVEL.isel(
                    time=slice(t_start, t_end), Z=slice(d_start, d_end)
                ).values
                vvel_chunk = ds.VVEL.isel(
                    time=slice(t_start, t_end), Z=slice(d_start, d_end)
                ).values
                wvel_chunk = ds.WVEL.isel(
                    time=slice(t_start, t_end), Zl=slice(d_start, d_end)
                ).values
                theta_chunk = ds.THETA.isel(
                    time=slice(t_start, t_end), Z=slice(d_start, d_end)
                ).values
                total_load_time = time.time() - load_start

                mem_after_load = process.memory_info().rss / 1024**3
                delta = mem_after_load - mem_before
                if verbose:
                    print(
                        f"[Task {task_id}] Worker {worker_id} AFTER: {mem_after_load:.2f} GB (delta: {delta:.2f} GB) | Time [{t_start}:{t_end}], Depth [{d_start}:{d_end}]",
                        flush=True,
                    )
                    # Flag spikes
                    if delta > 3.0:
                        print(
                            f"⚠️ [SPIKE DETECTED] Task {task_id}: Time [{t_start}:{t_end}], Depth [{d_start}:{d_end}] allocated {delta:.2f} GB",
                            flush=True,
                        )

                # Calculate expected memory from actual array sizes
                expected_memory_gb = (
                    uvel_chunk.nbytes
                    + vvel_chunk.nbytes
                    + wvel_chunk.nbytes
                    + theta_chunk.nbytes
                ) / 1e9

                if verbose:
                    # print(f"[Task {task_id}] Worker {worker_id} memory AFTER loading: {mem_after_load:.2f} GB")
                    print(
                        f"[Task {task_id}] Worker {worker_id} data should be ~{expected_memory_gb:.2f} GB"
                    )
                    print(
                        f"[Task {task_id}] Worker {worker_id} memory increase: {mem_after_load - mem_before:.2f} GB"
                    )

                # Time computation
                compute_start = time.time()
                chunk_results = []

                for ti_idx, ti in enumerate(range(t_start, t_end)):
                    for di_idx, di in enumerate(range(d_start, d_end)):
                        # Extract from already-loaded chunk
                        uvel = uvel_chunk[ti_idx, di_idx].T
                        vvel = vvel_chunk[ti_idx, di_idx].T
                        wvel = wvel_chunk[ti_idx, di_idx].T
                        theta = theta_chunk[ti_idx, di_idx].T

                        date = pd.to_datetime(
                            swirl_input_data.times[ti]
                        ).to_pydatetime()
                        depth = float(swirl_input_data.depths[di])
                        dz = swirl_input_data.dz_array[di]

                        result = run_parallel_task(
                            uvel,
                            vvel,
                            wvel,
                            theta,
                            swirl_input_data.dx,
                            swirl_input_data.dy,
                            dz,
                            pp_config["swirl_params_path"],
                            date,
                            depth,
                            ti,
                            di,
                            pp_config["output_folder"],
                            max_n_evc_points=max_n_evc_points,
                            overhead_factor=overhead_factor,
                            verbose=verbose,
                        )

                        chunk_results.append(result)

                total_compute_time = time.time() - compute_start

                # After deletion
                del uvel_chunk, vvel_chunk, wvel_chunk, theta_chunk
                gc.collect()
                gc.collect()
                gc.collect()

                if HAS_MALLOC_TRIM:
                    import ctypes
                    import ctypes.util

                    libc_name = ctypes.util.find_library("c")
                    if libc_name:
                        libc = ctypes.CDLL(libc_name)
                        libc.malloc_trim(0)

                mem_after_delete = process.memory_info().rss / 1024**3
                if verbose:
                    print(
                        f"[Task {task_id}] Worker {worker_id} memory AFTER delete: {mem_after_delete:.2f} GB"
                    )

                # Combine results from this chunk
                if chunk_results:
                    combined = pd.concat(chunk_results, ignore_index=True)
                    combined["task_load_time_sec"] = total_load_time
                    combined["task_compute_time_sec"] = total_compute_time
                    combined["task_total_time_sec"] = (
                        total_load_time + total_compute_time
                    )
                    combined["pixels_in_task"] = len(chunk_results)
                    return combined
                else:
                    return pd.DataFrame(
                        columns=[
                            "task_load_time_sec",
                            "task_compute_time_sec",
                            "task_total_time_sec",
                            "pixels_in_task",
                        ]
                    )

            task_dict[task_id] = process_chunk(verbose=verbose)
            task_id += 1

    return task_dict


def main(config_path="config_postprocessing.json"):
    # Start total execution timer
    total_start_time = time.time()

    with open(config_path, "r") as f:
        pp_config = json.load(f)

    verbose = pp_config.get("verbose", False)
    nb_cores = get_number_of_cores(pp_config)

    # Setup SLURM-aware Dask client
    client, cluster = setup_dask_client_slurm(
        nb_cores, memory_per_worker="auto", verbose=verbose
    )

    try:
        # ---------------------------------
        print(f"Loading input data... ({get_str_current_time()})")
        if pp_config["mitgcm_output_format"] == "netcdf":
            grid_nc_path = os.path.join(
                pp_config["grid_folder"], "merged_grid.nc"
            )
            swirl_input_data = load_input_data_netcdf(
                pp_config["i_mitgcm_folder_path"],
                grid_nc_path,
                pp_config["px"],
                pp_config["py"],
            )
        elif pp_config["mitgcm_output_format"] == "binary":
            swirl_input_data = load_input_data_binary(
                pp_config["i_mitgcm_folder_path"],
                pp_config["binary_mitgcm_grid_folder_path"],
                pp_config["binary_ref_date"],
                pp_config["binary_dt"],
            )
        else:
            raise ValueError(
                f'"mitgcm_output_format" must be either "netcdf" or "binary": {pp_config["mitgcm_output_format"]}'
            )

        start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime("%Y%m%d")
        end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime("%Y%m%d")

        # --------------------------------
        # Optionally save merged MITgcm results (be cautious with memory)
        if pp_config.get("save_nc_mitgcm", False):
            print(f"Reformatting input data... ({get_str_current_time()})")
            ds_reformat = reformat_mitgcm_results(
                swirl_input_data.ds_mitgcm,
                swirl_input_data.times,
                swirl_input_data.depths,
                pp_config["grid_folder"],
                nodata=np.nan,
            )

            print(f"Saving merged mitgcm results... ({get_str_current_time()})")
            mitgcm_output_folder = os.path.join(
                pp_config["output_folder"], O_MITGCM_FOLDER_NAME
            )
            os.makedirs(mitgcm_output_folder, exist_ok=True)
            output_path = os.path.join(
                mitgcm_output_folder, f"mitgcm_{start_date_str}_{end_date_str}.nc"
            )
            save_to_netcdf(ds_reformat, output_path)

        # ---------------------------------
        ds = swirl_input_data.ds_mitgcm

        print(f"Computing lake characteristics... ({get_str_current_time()})")
        ke_lake = []

        # Process lake characteristics in batches to control memory
        n_times = len(swirl_input_data.times)
        n_depths = len(swirl_input_data.depths)
        batch_size_time = estimate_batch_size(
            ds,
            n_times,
            n_depths,
            nb_cores=nb_cores,
            safety_factor=0.3,
            account_for_overhead=True,
            target_tasks_multiplier=1,
        )

        # Create delayed tasks for batches
        lake_tasks = []

        for t_start in range(0, n_times, batch_size_time):
            t_end = min(t_start + batch_size_time, n_times)

            @dask.delayed
            def compute_lake_batch(t_start=t_start, t_end=t_end):
                """Process a batch of time steps"""
                batch_results = []

                # Load entire batch at once (efficient!)
                uvel_batch = ds.UVEL.isel(time=slice(t_start, t_end)).values
                vvel_batch = ds.VVEL.isel(time=slice(t_start, t_end)).values
                wvel_batch = ds.WVEL.isel(time=slice(t_start, t_end)).values
                theta_batch = ds.THETA.isel(time=slice(t_start, t_end)).values

                # Process each (time, depth) in the batch
                for ti_batch, ti_global in enumerate(range(t_start, t_end)):
                    date = pd.to_datetime(
                        swirl_input_data.times[ti_global]
                    ).to_pydatetime()
                    for di, depth_val in enumerate(swirl_input_data.depths):
                        u = uvel_batch[ti_batch, di].T
                        v = vvel_batch[ti_batch, di].T
                        w = wvel_batch[ti_batch, di].T
                        theta = theta_batch[ti_batch, di].T

                        ke_slice = compute_ke_snapshot(
                            u,
                            v,
                            w,
                            swirl_input_data.dx,
                            swirl_input_data.dy,
                            dz=swirl_input_data.dz_array[di],
                        ).sum()
                        mean_temperature = theta[theta > 0].mean()
                        surface = (
                            len(theta[theta > 0])
                            * swirl_input_data.dx
                            * swirl_input_data.dy
                        )
                        volume = surface * swirl_input_data.dz_array[di]

                        batch_results.append(
                            {
                                "time_index": ti_global,
                                "depth_index": di,
                                "date": date,
                                "depth_[m]": depth_val,
                                "surface_area_[m2]": surface,
                                "volume_slice_[m3]": volume,
                                "kinetic_energy_[MJ]": ke_slice,
                                "mean_temperature_lake_[°C]": mean_temperature,
                            }
                        )

                # CRITICAL: Delete large arrays before returning
                del uvel_batch, vvel_batch, wvel_batch, theta_batch
                gc.collect()

                return batch_results

            lake_tasks.append(compute_lake_batch())

        print(f"  Created {len(lake_tasks)} lake characteristic tasks")
        print(f"  Computing in parallel...")

        # Compute all batches in parallel
        lake_results = dask.compute(*lake_tasks)

        # Flatten list of lists into single list
        ke_lake = [item for batch in lake_results for item in batch]

        print(f"Saving lake characteristics... ({get_str_current_time()})")
        lvl0_out_dir = os.path.join(pp_config["output_folder"], O_LVL0_FOLDER_NAME)
        os.makedirs(lvl0_out_dir, exist_ok=True)
        lake_csv_output_path = os.path.join(
            lvl0_out_dir, f"lake_characteristics_{start_date_str}_{end_date_str}.csv"
        )
        pd.DataFrame(ke_lake).to_csv(lake_csv_output_path, index=False)

        print(f"Preparing parallel tasks... ({get_str_current_time()})")

        # Configure chunking (add to config file or hardcode)
        time_chunk_size = pp_config.get("time_chunk_size", 1)  # Default: no chunking
        depth_chunk_size = pp_config.get("depth_chunk_size", 1)  # Default: no chunking

        # Create chunked tasks
        task_dict = create_chunked_tasks(
            swirl_input_data,
            pp_config,
            ds,
            nb_cores,
            time_chunk_size=time_chunk_size,
            depth_chunk_size=depth_chunk_size,
            target_tasks_multiplier=pp_config.get("target_tasks_multiplier", 3.0),
            max_n_evc_points=pp_config.get("max_n_evc_points", 30000),
            overhead_factor=pp_config.get("overhead_factor", 1.25),
            verbose=verbose,
        )

        # ---------------------------------
        print(f"Computing parallel tasks... ({get_str_current_time()})")

        from dask.distributed import as_completed
        from tqdm import tqdm

        max_in_flight = nb_cores
        task_iter = iter(task_dict.items())
        futures_to_keys = {}

        # Start initial batch
        for _ in range(max_in_flight):
            try:
                key, task = next(task_iter)
                fut = client.compute(task)
                futures_to_keys[fut] = key
            except StopIteration:
                break

        results = {}
        pbar = tqdm(total=len(task_dict))

        while futures_to_keys:
            # as_completed over current futures
            for fut in as_completed(futures_to_keys):
                key = futures_to_keys.pop(fut)
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = e
                pbar.update(1)

                # Submit the next task if available
                try:
                    key, task = next(task_iter)
                    new_fut = client.compute(task)
                    futures_to_keys[new_fut] = key
                except StopIteration:
                    pass

        pbar.close()

        print(f"Finished all tasks. Concatenating and saving to csv... ({get_str_current_time()})")

        if not results:
            raise ValueError("No results were computed successfully!")

        # Extract timing information (one measurement per TASK)
        task_load_times = []
        task_compute_times = []
        task_total_times = []
        pixels_per_task = []
        empty_results = 0

        for key, result_df in results.items():
            # Check if DataFrame is empty
            if result_df.empty:
                empty_results += 1
                continue

            # Each result is a DataFrame, extract timing from first row (should be same for all rows in the DataFrame)
            # Check if timing columns exist and extract TASK timing
            if (
                "task_load_time_sec" in result_df.columns
                and "task_compute_time_sec" in result_df.columns
            ):
                load_time = result_df["task_load_time_sec"].iloc[
                    0
                ]  # Same for all rows in this task
                compute_time = result_df["task_compute_time_sec"].iloc[0]
                total_time = result_df["task_total_time_sec"].iloc[0]
                n_pixels = (
                    result_df["pixels_in_task"].iloc[0]
                    if "pixels_in_task" in result_df.columns
                    else len(result_df)
                )

                task_load_times.append(load_time)
                task_compute_times.append(compute_time)
                task_total_times.append(total_time)
                pixels_per_task.append(n_pixels)

        # Report any issues
        if empty_results > 0:
            print(f"⚠️  Warning: {empty_results} tasks returned empty results")

        # Calculate statistics
        if task_total_times:
            avg_load = np.mean(task_load_times)
            avg_compute = np.mean(task_compute_times)
            avg_total = np.mean(task_total_times)

            min_load = np.min(task_load_times)
            max_load = np.max(task_load_times)

            min_compute = np.min(task_compute_times)
            max_compute = np.max(task_compute_times)

            min_total = np.min(task_total_times)
            max_total = np.max(task_total_times)

            std_total = np.std(task_total_times)

            avg_pixels = np.mean(pixels_per_task)
            total_pixels = sum(pixels_per_task)

            # Calculate overhead percentage
            overhead_pct = (avg_load / avg_total) * 100 if avg_total > 0 else 0

            # Calculate per-pixel timing
            avg_time_per_pixel = avg_total / avg_pixels if avg_pixels > 0 else 0
            avg_load_per_pixel = avg_load / avg_pixels if avg_pixels > 0 else 0
            avg_compute_per_pixel = avg_compute / avg_pixels if avg_pixels > 0 else 0

            print("\n" + "=" * 60)
            print("TASK TIMING ANALYSIS")
            print("=" * 60)
            print(f"Total tasks:           {len(results)}")
            print(f"Tasks with timing:     {len(task_total_times)}")
            print(f"Total pixels:          {total_pixels}")
            print(f"Avg pixels per task:   {avg_pixels:.1f}")

            print(f"\nTask Duration (per task - entire chunk):")
            print(f"  Data Loading:")
            print(f"    Average: {avg_load:.2f} sec")
            print(f"    Min:     {min_load:.2f} sec")
            print(f"    Max:     {max_load:.2f} sec")
            print(f"  Computation:")
            print(f"    Average: {avg_compute:.2f} sec")
            print(f"    Min:     {min_compute:.2f} sec")
            print(f"    Max:     {max_compute:.2f} sec")
            print(f"  Total:")
            print(f"    Average: {avg_total:.2f} sec")
            print(f"    Min:     {min_total:.2f} sec")
            print(f"    Max:     {max_total:.2f} sec")
            print(f"    Std Dev: {std_total:.2f} sec")

            print(f"\nPer-Pixel Performance:")
            print(f"  Loading:     {avg_load_per_pixel:.3f} sec/pixel")
            print(f"  Computation: {avg_compute_per_pixel:.3f} sec/pixel")
            print(f"  Total:       {avg_time_per_pixel:.3f} sec/pixel")

            print(f"\nOverhead Analysis:")
            print(f"  Data loading overhead: {overhead_pct:.1f}% of total task time")

            # Load balancing analysis
            if max_total > 0:
                load_imbalance = (max_total - min_total) / max_total * 100
                print(f"\nLoad Balancing:")
                print(f"  Variation: {load_imbalance:.1f}% (min to max)")
                if load_imbalance > 50:
                    print(f"     HIGH variation - some tasks much slower than others")

            # Recommendation
            print(f"\n{'RECOMMENDATION':^60}")
            print("-" * 60)

            # Task duration recommendations
            if avg_total < 5:
                print("   Tasks are VERY SHORT (< 5 sec)")
                print("   → Increase chunking to make tasks longer")
                print("   → Suggested: Multiply current chunk sizes by 2-3x")
            elif avg_total < 15:
                print("   Tasks are SHORT (5-15 sec)")
                print("   → Consider increasing chunk sizes slightly")
                print("   → Suggested: Multiply current chunk sizes by 1.5-2x")
            elif avg_total < 60:
                print("   Tasks are GOOD (15-60 sec)")
                print("   → Current chunking is well-balanced")
            else:
                print("   Tasks are LONG (> 60 sec)")
                print("   → Consider reducing chunk sizes for better load balancing")
                print("   → Suggested: Divide current chunk sizes by 1.5-2x")

            # Overhead recommendations
            if overhead_pct > 30:
                print(f"\n   HIGH DATA LOADING OVERHEAD: {overhead_pct:.1f}% of time!")
                print("   → Increasing chunk sizes will reduce relative overhead")
            elif overhead_pct > 15:
                print(
                    f"\n   MODERATE overhead: {overhead_pct:.1f}% spent on data loading"
                )
                print("   → Chunking is helping, but could be improved")
            else:
                print(
                    f"\n✓  LOW overhead: Only {overhead_pct:.1f}% spent on data loading"
                )

            # Load balancing recommendations
            if load_imbalance > 50 and len(task_total_times) < 50:
                print(f"\n   High load imbalance with few tasks!")
                print("   → Decrease chunk sizes to create more tasks")
                print("   → This improves load balancing across workers")

            # Efficiency note
            total_computation_time = sum(task_total_times)
            n_workers = pp_config.get("px", 1) * pp_config.get("py", 1)
            ideal_parallel_time = total_computation_time / n_workers
            print(f"\nEfficiency Estimate:")
            print(f"  Total computation: {total_computation_time:.1f} sec")
            print(
                f"  Ideal parallel time (perfect efficiency): {ideal_parallel_time:.1f} sec"
            )
            print(f"  With {n_workers} workers")

            print("=" * 60 + "\n")
        else:
            print("   No timing information found in results")
            print("   Make sure timing is properly added in create_chunked_tasks()")

        # Continue with concatenation
        print(f"Concatenating and saving to csv... ({get_str_current_time()})")
        df_catalogue_level0 = pd.concat(
            [results[key] for key in sorted(results.keys())], ignore_index=True
        )

        # ---------------------------------
        # Prepare output folder + CSV
        output_path = os.path.join(
            lvl0_out_dir, f"lvl0_{start_date_str}_{end_date_str}.csv"
        )
        df_catalogue_level0.to_csv(output_path, index=False)

        # ---------------------------------
        print(f"Checking if MITgcm diverged... ({get_str_current_time()})")
        last_theta_slice = ds.THETA.isel(time=-1, Z=0).values
        if np.all(np.isnan(last_theta_slice)):
            raise ValueError(
                "Last time step of MITgcm results contains only NaN values."
            )

        # ---------------------------------
        print(f"Done. ({get_str_current_time()})")

        # Print total execution time
        total_elapsed = time.time() - total_start_time
        hours = int(total_elapsed // 3600)
        minutes = int((total_elapsed % 3600) // 60)
        seconds = total_elapsed % 60

        print("\n" + "=" * 60)
        print(f"TOTAL EXECUTION TIME: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        print(f"  ({total_elapsed:.2f} seconds)")
        print("=" * 60)

    except Exception as e:
        # Calculate elapsed time even on error
        total_elapsed = time.time() - total_start_time
        hours = int(total_elapsed // 3600)
        minutes = int((total_elapsed % 3600) // 60)
        seconds = total_elapsed % 60

        print(f"\nFatal error in main: {e}")
        print(f"Execution time before error: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        traceback.print_exc()
        raise

    finally:
        # Always close the client
        print("Shutting down Dask client...")
        client.close()
        cluster.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    # multiprocessing compatibility for Windows (harmless on Linux)
    from multiprocessing import freeze_support

    freeze_support()
    main('config_geneva_50m.json')
