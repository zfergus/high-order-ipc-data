#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --mem=64GB

# Load modules
module purge
module load cmake/3.22.2 python/intel/3.8.6 gcc/10.2.0 hdf5/intel/1.12.0

SCRIPTS_ROOT=$SCRATCH/decoupled-contact/scripts
SCRIPT=$(realpath $1)
SCRIPT_REL=$(realpath --relative-to=$SCRIPTS_ROOT $SCRIPT)

OUTPUT_ROOT=$SCRATCH/decoupled-contact/results

# Drop the extension from script
OUTPUT_DIR="$OUTPUT_ROOT/${SCRIPT_REL%.*}"

# Run job
cd $SCRATCH/polyfem/build/release
./PolyFEM_bin -j $SCRIPT -o $OUTPUT_DIR --ngui --log_level debug
