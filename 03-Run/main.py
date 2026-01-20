"""
Created on 30.07.2025 by Anne Leroquais (EAWAG) - memory-safe dask version
Modified on 08.10.2025 by Flavio Calvo (UNIL), with great help from Anthropic/Sonnet 4.5 (code generation) and black (code formatting)

Notes:
- Expects load_input_data_netcdf / load_input_data_binary to return lazy xarray DataArrays
  chunked so that .isel(time=..., Z=...) returns a small Dask graph (1 plane).
"""

import os
import sys
import glob
import re
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
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, better for saving files
import matplotlib.pyplot as plt

from functions import *

O_MITGCM_FOLDER_NAME = "mitgcm_results"
O_GRID_FOLDER_NAME = "grid"
O_LVL0_FOLDER_NAME = "eddy_catalogues_lvl0"
O_FIGURE_FOLDER_NAME = "figures"

DEFAULT_CONFIG_NAME = "config_lucerne.json"


def can_use_malloc_trim():
    """Check if malloc_trim is available (cached at module level)"""
    libc_name = ctypes.util.find_library("c")
    return libc_name is not None


# Use malloc_trim to aggressively return freed memory (with the GNU allocator only)
HAS_MALLOC_TRIM = can_use_malloc_trim()
print(f"malloc_trim available: {HAS_MALLOC_TRIM}")


def get_str_current_time():
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")


def compute_eddy_catalogue(
        u, v, theta,
        ke_slice,
        dx, dy, dz,
        swirl_params_path,
        date,  depth,
        t_index, d_index,
        output_folder,
        id_level0=0,
        verbose=False,
):
    # Eddy detection & feature extraction (same as before)
    if verbose:
        print(f"Run swirl for t={t_index}, d={d_index}")
        print(f"[TASK] About to call run_swirl...", flush=True)

    eddies = run_swirl(
        u, v,
        dx, dy,
        swirl_params_path,
    )

    if verbose:
        print(f"[TASK] run_swirl RETURNED!", flush=True)

    if hasattr(eddies, "n_vortices") and eddies.n_vortices == 0:
        if verbose:
            print(f"[TASK] No vortices found, returning empty", flush=True)
        return pd.DataFrame()

    if verbose:
        print(f"[TASK] Processing {eddies.n_vortices} vortices...", flush=True)

    eddy_rows = []
    for eddy_index, eddy in enumerate(eddies):
        indices_eddy = (t_index, d_index, eddy_index)
        row_data = extract_eddy_data(
            indices_eddy, eddy, date, depth, dz, ke_slice, dx * dy, theta, id_level0
        )
        eddy_rows.append(row_data)

    # Optional map output for first depth
    if d_index == 0:
        if verbose:
            print(f"Map t={t_index}, d={d_index}")
        fig = plot_map_swirl(u.T, v.T, eddies, date, 6)
        fig_path = os.path.join(output_folder, O_FIGURE_FOLDER_NAME, f"map_swirl_{date}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # IMPORTANT: free matplotlib memory
        plt.close("all")  # Close all figures

    del u, v, theta, ke_slice,
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
        nb_cores = int(pp_config.get("cores", 1))
        if verbose:
            print(
                f"No SLURM detected. Using config file: = {nb_cores} cores"
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


    swirl_safety_buffer = 0
    total_memory_per_timestep = memory_per_timestep

    if verbose:
        print(
            f"  Data types: UVEL={sample_uvel.dtype}, VVEL={sample_vvel.dtype}, WVEL={sample_wvel.dtype}, THETA={sample_theta.dtype}"
        )
        print(
            f"  Memory per timestep (input data): {memory_per_timestep / 1e9:.2f} GB"
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


def setup_dask_client(nb_cores, memory_per_worker="auto", verbose=False):
    """
    Setup Dask client with SLURM-aware memory limits.

    Parameters:
    -----------
    nb_cores : int
        Number of workers (typically from SLURM_CPUS_PER_TASK or pp_config["cores"])
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


def get_chunk_sizes(
        time_chunk_size, depth_chunk_size,
        ds,
        n_depths, n_times, nb_cores,
        target_tasks_multiplier,
        verbose=False):

    """
    Estimates the time and depth chunks sizes based on manual/auto values,
    available memory and number of available cores.

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

    :return: time_chunk_size, depth_chunk_size
    """

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

    return time_chunk_size, depth_chunk_size


def create_chunked_tasks(
    swirl_input_data,
    pp_config,
    ds,
    nb_cores,
    time_chunk_size=2,
    depth_chunk_size=50,
    target_tasks_multiplier=3,
    verbose=False,
):
    """
    Create chunked tasks to reduce overhead.

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

    if verbose:
        pixels_per_task = time_chunk_size * depth_chunk_size
        total_pixels = n_times * n_depths
        total_tasks = len(time_chunks) * len(depth_chunks)

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

                if verbose:
                    # Check what's in memory
                    all_objects = gc.get_objects()
                    large_objects = []
                    for obj in all_objects:
                        try:
                            size = sys.getsizeof(obj)
                            if size > 10 * 1024 * 1024:  # > 10 MB
                                large_objects.append((type(obj).__name__, size / 1024 ** 2))
                        except:
                            pass

                    large_objects.sort(key=lambda x: x[1], reverse=True)
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

                if verbose:
                    # Calculate expected memory from actual array sizes
                    expected_memory_gb = (
                                                 uvel_chunk.nbytes
                                                 + vvel_chunk.nbytes
                                                 + wvel_chunk.nbytes
                                                 + theta_chunk.nbytes
                                         ) / 1e9
                    # print(f"[Task {task_id}] Worker {worker_id} memory AFTER loading: {mem_after_load:.2f} GB")
                    print(
                        f"[Task {task_id}] Worker {worker_id} data should be ~{expected_memory_gb:.2f} GB"
                    )
                    print(
                        f"[Task {task_id}] Worker {worker_id} memory increase: {mem_after_load - mem_before:.2f} GB"
                    )

                # Time computation
                compute_start = time.time()
                eddy_results = []
                lake_results = []

                for ti_idx, ti in enumerate(range(t_start, t_end)):
                    for di_idx, di in enumerate(range(d_start, d_end)):
                        # Extract from already-loaded chunk
                        uvel = uvel_chunk[ti_idx, di_idx].T
                        vvel = vvel_chunk[ti_idx, di_idx].T
                        wvel = wvel_chunk[ti_idx, di_idx].T
                        theta = theta_chunk[ti_idx, di_idx].T

                        date = pd.to_datetime(swirl_input_data.times[ti]).to_pydatetime()
                        depth = float(swirl_input_data.depths[di])
                        dz = swirl_input_data.dz_array[di]

                        # First, compute lake characteristics
                        ke_slice = compute_ke_snapshot(
                            uvel,
                            vvel,
                            wvel,
                            swirl_input_data.dx,
                            swirl_input_data.dy,
                            dz=swirl_input_data.dz_array[di],
                        )
                        temperature_slice = theta[theta > 0]
                        surface = (
                                len(theta[theta > 0])
                                * swirl_input_data.dx
                                * swirl_input_data.dy
                        )
                        volume = surface * swirl_input_data.dz_array[di]
                        lake_results.append(pd.DataFrame([{
                                "time_index": ti,
                                "depth_index": di,
                                "date": date,
                                "depth_[m]": depth,
                                "surface_area_[m2]": surface,
                                "volume_slice_[m3]": volume,
                                "kinetic_energy_[MJ]": ke_slice.sum(),
                                "mean_temperature_lake_[°C]": temperature_slice.mean(),
                            }]))

                        # Then, compute eddies
                        eddy_catalogue = compute_eddy_catalogue(
                            uvel, vvel, theta,
                            ke_slice,
                            swirl_input_data.dx,
                            swirl_input_data.dy,
                            dz,
                            pp_config["swirl_params_path"],
                            date, depth,
                            ti, di,
                            pp_config["output_folder"],
                            verbose=verbose,
                        )
                        eddy_results.append(eddy_catalogue)



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

                if verbose:
                    mem_after_delete = process.memory_info().rss / 1024 ** 3
                    print(
                        f"[Task {task_id}] Worker {worker_id} memory AFTER delete: {mem_after_delete:.2f} GB"
                    )

                # Combine results from this chunk
                combined_lake_results = pd.DataFrame()
                if lake_results:
                    combined_lake_results = pd.concat(lake_results, ignore_index=True)

                combined_eddy_results = pd.DataFrame(
                        columns=[
                            "task_load_time_sec",
                            "task_compute_time_sec",
                            "task_total_time_sec",
                            "pixels_in_task",
                        ]
                    )
                if eddy_results:
                    combined_eddy_results = pd.concat(eddy_results, ignore_index=True)

                    if verbose:
                        combined_eddy_results["task_load_time_sec"] = total_load_time
                        combined_eddy_results["task_compute_time_sec"] = total_compute_time
                        combined_eddy_results["task_total_time_sec"] = (
                            total_load_time + total_compute_time
                        )
                        combined_eddy_results["pixels_in_task"] = len(eddy_results)

                return (combined_lake_results, combined_eddy_results)

            task_dict[task_id] = process_chunk(verbose=verbose)
            task_id += 1

    return task_dict


def print_run_statistics(results, pp_config, total_start_time):
    # Extract timing information (one measurement per TASK)
    task_load_times = []
    task_compute_times = []
    task_total_times = []
    pixels_per_task = []
    empty_results = 0

    for result_df in results:
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
        n_workers = pp_config.get("cores", 1)
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

    # Print total execution time
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = total_elapsed % 60

    print("\n" + "=" * 60)
    print(f"TOTAL EXECUTION TIME: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    print(f"  ({total_elapsed:.2f} seconds)")
    print("=" * 60)


def compute_parallel_tasks(nb_cores, task_dict, client):
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

    return results


def load_input_data(pp_config, iter_number):
    if pp_config["mitgcm_output_format"] == "netcdf":
        grid_nc_path = os.path.join(
            pp_config.get("grid_folder", ""), "merged_grid.nc"
        )
        swirl_input_data = load_input_data_netcdf(
            pp_config.get("i_mitgcm_folder_path", ""),
            grid_nc_path,
            pp_config.get("nc_px", 1),
            pp_config.get("nc_py", 1),
            idx_z_cut=pp_config.get("idx_z_cut", None)
        )
    elif pp_config["mitgcm_output_format"] == "binary":
        print(iter_number)
        swirl_input_data = load_input_data_binary(
            pp_config.get("i_mitgcm_folder_path"),
            pp_config.get("binary_mitgcm_grid_folder_path"),
            pp_config.get("binary_ref_date"),
            pp_config.get("binary_dt"),
            iter_numbers=iter_number,
            idx_z_cut=pp_config.get("idx_z_cut", None),
        )
    else:
        raise ValueError(
            f'"mitgcm_output_format" must be either "netcdf" or "binary": {pp_config["mitgcm_output_format"]}'
        )

    return swirl_input_data


def run_postprocessing_for_one_iteration(pp_config, iter_number, nb_cores, verbose, client, total_start_time):
    """

    :param pp_config:
    :param iter_number:
    :param nb_cores:
    :param verbose:
    :param client:
    :param total_start_time:
    :return: ds
    """
    # ---------------------------------
    print(f"Loading input data for iteration {iter_number}... ({get_str_current_time()})")
    swirl_input_data = load_input_data(pp_config, iter_number)

    start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime("%Y%m%d")
    end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime("%Y%m%d")

    # ---------------------------------
    print(f"Preparing parallel tasks... ({get_str_current_time()})")

    # Estimate chunk size
    time_chunk_size, depth_chunk_size = get_chunk_sizes(
        pp_config.get("time_chunk_size", 1),
        pp_config.get("depth_chunk_size", 1),
        swirl_input_data.ds_mitgcm,
        len(swirl_input_data.depths),
        len(swirl_input_data.times),
        nb_cores,
        pp_config.get("target_tasks_multiplier", 3.0),
        verbose=False)

    # Create chunked tasks
    task_dict = create_chunked_tasks(
        swirl_input_data,
        pp_config,
        swirl_input_data.ds_mitgcm,
        nb_cores,
        time_chunk_size=time_chunk_size,
        depth_chunk_size=depth_chunk_size,
        target_tasks_multiplier=pp_config.get("target_tasks_multiplier", 3.0),
        verbose=verbose,
    )

    # ---------------------------------
    print(f"Computing parallel tasks... ({get_str_current_time()})")
    results = compute_parallel_tasks(nb_cores, task_dict, client)

    print(f"Finished all tasks. Concatenating and saving to csv... ({get_str_current_time()})")

    valid_results = [
        v for v in results.values() if isinstance(v, (tuple, pd.Series)) and len(v) > 0
    ]

    if not valid_results:
        raise ValueError("No valid DataFrames found in results!")

    # Extract and concat dataframes for lake and eddies
    lake_results = [v[0] for v in valid_results]
    eddy_results = [v[1] for v in valid_results]

    df_lake = pd.concat(lake_results, ignore_index=True)
    df_catalogue_level0 = pd.concat(eddy_results, ignore_index=True)

    # ---------------------------------
    # Prepare output folder + CSV
    csv_output_folder = os.path.join(pp_config.get("output_folder"), O_LVL0_FOLDER_NAME)
    df_lake.to_csv(
        os.path.join(csv_output_folder, f"lake_characteristics_{start_date_str}_{end_date_str}_{iter_number}.csv"),
        index=False)
    df_catalogue_level0.to_csv(
        os.path.join(csv_output_folder, f"lvl0_{start_date_str}_{end_date_str}_{iter_number}.csv"),
        index=False)

    print_run_statistics(eddy_results, pp_config, total_start_time)

    return swirl_input_data.ds_mitgcm


def reformat_and_save_ds_to_netcdf(pp_config):
    swirl_input_data = load_input_data(pp_config, "all")
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
    start_date_str = pd.Timestamp(swirl_input_data.times[0]).strftime("%Y%m%d")
    end_date_str = pd.Timestamp(swirl_input_data.times[-1]).strftime("%Y%m%d")
    output_path = os.path.join(
        mitgcm_output_folder, f"mitgcm_{start_date_str}_{end_date_str}.nc"
    )
    save_to_netcdf(ds_reformat, output_path)


def get_iter_numbers_from_filename(file_name:str):
    # Extract all numbers inside the square brackets
    numbers = re.findall(r'(?<=\[)[0-9, ]+(?=\])', file_name)
    nums=None
    if numbers:
        # Split by comma and convert each to int
        nums = [int(n.strip()) for n in numbers[0].split(',')]
    else:
        print("No numbers found.")

    return nums


def concat_and_save_csv(pp_config, prefix: str):
    out_dir = os.path.join(pp_config["output_folder"], O_LVL0_FOLDER_NAME)
    fname_base = os.path.join(out_dir, f"{prefix}_")
    flist = glob.glob(fname_base + "*.csv")
    dfs = []
    for file in flist:
        iter_numbers = get_iter_numbers_from_filename(file)
        if iter_numbers is None:
            continue
        df_temp = pd.read_csv(file)

        iter_idx = 0.5 + min(iter_numbers) * pp_config["binary_dt"] / pp_config["dt_save"]
        df_temp['time_index'] = df_temp['time_index'] + iter_idx

        dfs.append(df_temp)

    df_concat = pd.concat(dfs, ignore_index=True).sort_values(
        by=['time_index', 'depth_index']).reset_index(drop=True)
    df_concat['id'] = np.arange(len(df_concat))

    start_date_str = pd.Timestamp(df_concat['date'][0]).strftime("%Y%m%d")
    end_date_str = pd.Timestamp(df_concat['date'][len(df_concat)-1]).strftime("%Y%m%d")

    output_path = os.path.join(
        out_dir, f"{prefix}_{start_date_str}_{end_date_str}_concat.csv"
    )
    df_concat.to_csv(output_path, index=False)


def main(config_path="config_postprocessing.json"):
    # Start total execution timer
    total_start_time = time.time()

    with open(config_path, "r") as f:
        pp_config = json.load(f)

    output_folder = pp_config.get("output_folder")
    print(f"Creating output folders in: {output_folder}")
    os.makedirs(os.path.join(output_folder, O_FIGURE_FOLDER_NAME), exist_ok=True)
    os.makedirs(os.path.join(output_folder, O_MITGCM_FOLDER_NAME), exist_ok=True)
    os.makedirs(os.path.join(output_folder, O_LVL0_FOLDER_NAME), exist_ok=True)

    verbose = pp_config.get("verbose", False)
    nb_cores = get_number_of_cores(pp_config)

    # Setup SLURM-aware Dask client
    client, cluster = setup_dask_client(
        nb_cores, memory_per_worker="auto", verbose=verbose
    )

    try:
        # --------------------------------
        # Optionally save merged MITgcm results (be cautious with memory)
        if pp_config.get("save_nc_mitgcm", False):
            reformat_and_save_ds_to_netcdf(pp_config)

        if pp_config['iterations'] == "all":
            fname_base = os.path.join(pp_config["i_mitgcm_folder_path"], "3Dsnaps")
            flist = glob.glob(fname_base + "*.data")
            flist = [os.path.basename(f) for f in flist]
            if flist == []:
                raise ValueError(f"No files were found with path {fname_base + '*.data'}")
            iters = np.sort([int(file_name.split('.')[1]) for file_name in flist])
        else:
            iters = pp_config['iterations']

        print(f"iterations: ")
        print(iters)
        for i in range(0, len(iters), pp_config.get("time_chunk_size", 1)):
            print(f"\033[1;31mRunning iteration {i} out of {len(iters)}.\033[0m")
            iter_numbers = iters[i : i + pp_config.get("time_chunk_size", 1)]
            run_postprocessing_for_one_iteration(pp_config, list(iter_numbers), nb_cores, verbose, client, total_start_time)
            client.run(gc.collect)

        # ---------------------------------
        print(f"Finished all iterations. Concatenating and saving to final csv...")
        concat_and_save_csv(pp_config, "lvl0")
        concat_and_save_csv(pp_config, "lake_characteristics")

        # ---------------------------------
        print(f"Done. ({get_str_current_time()})")

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
    main(DEFAULT_CONFIG_NAME)
