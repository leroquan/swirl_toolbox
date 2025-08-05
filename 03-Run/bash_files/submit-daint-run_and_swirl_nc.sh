#!/bin/bash -l

#SBATCH --account=xx
#SBATCH --job-name="MITgcmInference"
#SBATCH --time=01:00:00
#SBATCH --ntasks=576
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mail-user=xx
#SBATCH --mail-type=end
# Script to run the simulation on the CSSC cluster

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=0
source activate swirl_toolbox

NLOOPS=2
STEPDURATION=604800
SIMTIMESTEP=60

sed -i "s/pChkptFreq.*=.*/pChkptFreq=${STEPDURATION}/" data

for (( i=0; i<NLOOPS; i++ ))
do
	echo "=== Loop $i: Updating configuration at $(date '+%Y-%m-%d %H:%M:%S') ==="

	i_start=$((i * STEPDURATION))
	end_time=$((i_start + STEPDURATION))
	sed -i "s/startTime.*=.*/startTime=${i_start}/" data
	sed -i "s/endTime.*=.*/endTime=${end_time}/" data
	
	echo "=== Loop $i: Starting MITgcm at $(date '+%Y-%m-%d %H:%M:%S') ==="

	# Run MITgcm
	srun --ntasks=${SLURM_NTASKS} ./mitgcmuv

	echo "=== Loop $i: Postprocessing at $(date '+%Y-%m-%d %H:%M:%S') ==="

	if [ "$i" -eq 0 ]; then
		# we only do this once, it should always be the same grid anyway
		python $MITGCM_ROOTDIR/utils/python/MITgcmutils/scripts/gluemncbig -o "../output/grid/merged_grid.nc" "../run/grid*.nc" --many
	fi

	# Run your Dask-based Python postprocessing
	python ../postprocessing/main.py

	echo "=== Loop $i: Cleaning up at $(date '+%Y-%m-%d %H:%M:%S') ==="

	# Keep the last pickup file, delete others
	find . -name "grid*.nc" -delete
	find . -name "3Dsnaps*.nc" -delete
	find . -name "monitor*.nc" -delete
	find . -name "phiHyd*.nc" -delete
	find . -name "state*.nc" -delete
	
	pickup_suff=$((end_time / SIMTIMESTEP))
	sed -i "s/pickupSuff.*=.*/pickupSuff=${pickup_suff}/" data
done