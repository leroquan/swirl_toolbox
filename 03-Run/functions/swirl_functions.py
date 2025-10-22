import swirl
import numpy as np

import os
import psutil
import time


def log_prefix(task_id=None, t_index=None, d_index=None, worker_id=None):
    """Generate consistent log prefix with task/worker/time/depth info"""
    import os

    if worker_id is None:
        worker_id = os.getpid()

    parts = [f"W{worker_id}"]

    if task_id is not None:
        parts.append(f"T{task_id[:6]}")  # Short task ID

    if t_index is not None:
        parts.append(f"t{t_index}")

    if d_index is not None:
        parts.append(f"d{d_index}")

    return f"[{' | '.join(parts)}]"


def log(msg, task_id=None, t_index=None, d_index=None, worker_id=None):
    """Print with consistent prefix"""
    prefix = log_prefix(task_id, t_index, d_index, worker_id)
    print(f"{prefix} {msg}", flush=True)


def run_swirl(u_plot, v_plot, dx, dy, swirl_params_path, *args, **kwargs):
    try:
        vortices = swirl.Identification(
            v=[u_plot, v_plot],
            grid_dx=[dx, dy],
            param_file=swirl_params_path,
            verbose=False,
        )

        vortices.run()

        return vortices

    except Exception as e:
        print(f"[SWIRL ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        raise


def run_swirl_profiled(
    u_plot,
    v_plot,
    dx,
    dy,
    swirl_params_path,
    max_n_evc_points=30000,
    overhead_factor=1.25,
    t_index=None,
    d_index=None,
    task_id=None,
):
    """Run SWIRL with detailed memory profiling at each step"""
    import psutil
    import os
    import gc

    def plog(msg):
        """Local logging helper"""
        log(msg, task_id=task_id, t_index=t_index, d_index=d_index)

    process = psutil.Process(os.getpid())
    worker_id = os.getpid()

    def get_mem():
        return process.memory_info().rss / 1e9

    def plog(msg):
        """Local logging helper"""
        log(msg, task_id=task_id, t_index=t_index, d_index=d_index, worker_id=worker_id)

    plog("SWIRL PROFILE | Starting...")
    mem_start = get_mem()
    plog(f"  Memory at start: {mem_start:.2f} GB")

    # === INITIALIZATION ===
    plog("SWIRL PROFILE | Creating Identification object...")
    mem_before_init = get_mem()

    vortices = swirl.Identification(
        v=[u_plot, v_plot],
        grid_dx=[dx, dy],
        param_file=swirl_params_path,
        verbose=False,
    )

    mem_after_init = get_mem()
    plog(
        f"  After __init__: {mem_after_init:.2f} GB (delta: {mem_after_init - mem_before_init:+.2f} GB)"
    )

    # === STEP 1: RORTEX ===
    plog("SWIRL PROFILE | Computing rortex...")
    mem_before_rortex = get_mem()

    _ = vortices.rortex

    mem_after_rortex = get_mem()
    plog(
        f"  After rortex: {mem_after_rortex:.2f} GB (delta: {mem_after_rortex - mem_before_rortex:+.2f} GB)"
    )

    gc.collect()
    mem_after_gc1 = get_mem()
    if abs(mem_after_gc1 - mem_after_rortex) > 0.05:
        plog(
            f"  After GC: {mem_after_gc1:.2f} GB (freed: {mem_after_rortex - mem_after_gc1:.2f} GB)"
        )

    # === STEP 2: EVC MAP ===
    plog("SWIRL PROFILE | Computing EVC map...")
    mem_before_evc = get_mem()

    _ = vortices.gevc_map

    mem_after_evc = get_mem()
    plog(
        f"  After EVC map: {mem_after_evc:.2f} GB (delta: {mem_after_evc - mem_before_evc:+.2f} GB)"
    )

    gc.collect()
    mem_after_gc2 = get_mem()
    if abs(mem_after_gc2 - mem_after_evc) > 0.05:
        plog(
            f"  After GC: {mem_after_gc2:.2f} GB (freed: {mem_after_evc - mem_after_gc2:.2f} GB)"
        )

    # === ADDED: ANALYZE EVC POINTS BEFORE CLUSTERING ===
    try:
        # Access the EVC map (try different possible attribute names)
        gevc_map = None
        if hasattr(vortices, "_gevc_map") and vortices._gevc_map is not None:
            gevc_map = vortices._gevc_map
        elif hasattr(vortices, "gevc_map") and vortices.gevc_map is not None:
            gevc_map = vortices.gevc_map

        if gevc_map is not None and hasattr(gevc_map, "shape"):
            # gevc_map shape is (3, N) where:
            # Row 0: x coordinates
            # Row 1: y coordinates
            # Row 2: cardinality (if fast_clustering)

            n_evc_points = (
                gevc_map.shape[1] if len(gevc_map.shape) > 1 else len(gevc_map)
            )

            # Estimate distance matrix memory
            distance_matrix_gb = 2.0 * (n_evc_points**2) * 8 / 1e9

            plog(f"  EVC points detected: {n_evc_points}")
            plog(f"  Expected distance matrix size: {distance_matrix_gb:.2f} GB")

            # If fast_clustering, also check total cardinality
            if len(gevc_map.shape) > 1 and gevc_map.shape[0] >= 3:
                cardinality = gevc_map[2, :]  # Third row is cardinality
                stot = int(np.sum(np.abs(cardinality)))

                if stot > n_evc_points:  # Only log if cardinality matters
                    fast_clustering_gb = 2 * stot * 8 / 1e9
                    plog(f"  Total cardinality (fast_clustering): {stot}")
                    plog(
                        f"  Expected fast_clustering arrays: {fast_clustering_gb:.2f} GB"
                    )

                    total_expected = distance_matrix_gb + fast_clustering_gb
                    plog(f"  Total expected clustering memory: {total_expected:.2f} GB")

    except Exception as e:
        plog(f"  Warning: Could not analyze EVC points: {e}")

    # === ADDED: CHECK AND CROP EVC POINTS IF NEEDED ===
    try:
        # Access the EVC map
        gevc_map = None
        if hasattr(vortices, "_gevc_map") and vortices._gevc_map is not None:
            gevc_map = vortices._gevc_map
        elif hasattr(vortices, "gevc_map") and vortices.gevc_map is not None:
            gevc_map = vortices.gevc_map

        if gevc_map is not None and hasattr(gevc_map, "shape"):
            n_evc_points = (
                gevc_map.shape[1] if len(gevc_map.shape) > 1 else len(gevc_map)
            )

            # Estimate memory requirements
            distance_matrix_gb = overhead_factor * 2.0 * (n_evc_points**2) * 8 / 1e9

            plog(f"  EVC points detected: {n_evc_points}")
            plog(
                f"  Expected clustering memory: {distance_matrix_gb:.2f} GB (with {overhead_factor}× overhead)"
            )

            # ADDED: Check if cropping is needed
            if n_evc_points > max_n_evc_points:
                plog(
                    f"⚠️  WARNING: EVC points ({n_evc_points}) exceeds limit ({max_n_evc_points})"
                )
                plog(f"   Estimated memory would be {distance_matrix_gb:.2f} GB")

                # ADDED: Subsample by selecting points with highest absolute cardinality
                # gevc_map shape: (3, N) where row 2 is cardinality
                if len(gevc_map.shape) > 1 and gevc_map.shape[0] >= 3:
                    cardinality = np.abs(gevc_map[2, :])

                    # Get indices of top max_n_evc_points by cardinality
                    top_indices = np.argsort(cardinality)[-max_n_evc_points:]

                    # Subsample the gevc_map
                    gevc_map_cropped = gevc_map[:, top_indices]

                    # Update vortices object with cropped map
                    if hasattr(vortices, "_gevc_map"):
                        vortices._gevc_map = gevc_map_cropped
                    else:
                        vortices.gevc_map = gevc_map_cropped

                    plog(
                        f"   ✓ Subsampled to {max_n_evc_points} points with highest cardinality"
                    )

                    # Update memory estimate
                    new_distance_matrix_gb = (
                        overhead_factor * 2.0 * (max_n_evc_points**2) * 8 / 1e9
                    )
                    plog(
                        f"   New expected clustering memory: {new_distance_matrix_gb:.2f} GB"
                    )
                else:
                    plog(f"   ⚠️  Cannot subsample: gevc_map format unexpected")

    except Exception as e:
        plog(f"  Warning: Could not check/crop EVC points: {e}")

    # === STEP 3: CLUSTERING ===
    plog("SWIRL PROFILE | Running clustering...")
    mem_before_cluster = get_mem()

    vortices.clustering()

    mem_after_cluster = get_mem()
    plog(
        f"  After clustering: {mem_after_cluster:.2f} GB (delta: {mem_after_cluster - mem_before_cluster:+.2f} GB)"
    )

    gc.collect()
    mem_after_gc3 = get_mem()
    if abs(mem_after_gc3 - mem_after_cluster) > 0.05:
        plog(
            f"  After GC: {mem_after_gc3:.2f} GB (freed: {mem_after_cluster - mem_after_gc3:.2f} GB)"
        )

    # === STEP 4: DETECTION ===
    plog("SWIRL PROFILE | Running detection...")
    mem_before_detect = get_mem()

    vortices.detect_vortices()

    mem_after_detect = get_mem()
    plog(
        f"  After detection: {mem_after_detect:.2f} GB (delta: {mem_after_detect - mem_before_detect:+.2f} GB)"
    )

    gc.collect()
    mem_after_gc4 = get_mem()
    if abs(mem_after_gc4 - mem_after_detect) > 0.05:
        plog(
            f"  After GC: {mem_after_gc4:.2f} GB (freed: {mem_after_detect - mem_after_gc4:.2f} GB)"
        )

    # === SUMMARY ===
    mem_final = get_mem()
    plog(
        f"SWIRL PROFILE | Complete: {mem_final:.2f} GB (total delta: {mem_final - mem_start:+.2f} GB)"
    )
    plog(f"  Found {vortices.n_vortices} vortices")

    # === BREAKDOWN ===
    plog("SWIRL PROFILE | Memory deltas by step:")
    plog(f"  Init:       {mem_after_init - mem_before_init:+.2f} GB")
    plog(f"  Rortex:     {mem_after_rortex - mem_before_rortex:+.2f} GB")
    plog(f"  EVC map:    {mem_after_evc - mem_before_evc:+.2f} GB")
    plog(f"  Clustering: {mem_after_cluster - mem_before_cluster:+.2f} GB")
    plog(f"  Detection:  {mem_after_detect - mem_before_detect:+.2f} GB")
    plog(f"  Net change: {mem_final - mem_start:+.2f} GB")

    return vortices
