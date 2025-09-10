import dask


def save_to_netcdf(ds, output_path):
    print(f"Saving dataset")
    with dask.config.set(scheduler="single-threaded"):
        # Prepare encoding with proper chunksizes
        encoding = {}
        for var in ds.data_vars:
            if hasattr(ds[var].data, 'chunks'):  # Dask-backed array
                # Take the first chunk along each dimension
                encoding[var] = {"chunksizes": tuple(c[0] for c in ds[var].data.chunks)}
            else:
                encoding[var] = {}  # in-memory array, no chunking needed

        # Write dataset chunk by chunk
        ds.to_netcdf(
            output_path,
            format="NETCDF4",
            engine="netcdf4",
            compute=True,
            encoding=encoding
        )
