#!/bin/bash

# SBATCH --output=log_%j.out # name of output file
# SBATCH --error=log_%j.out  # name of error file (here, in common with the output file)
## GPU
# SBATCH -A vfw@v100
# SBATCH -C v100-32g
# SBATCH --nodes=1                    # we request one node
# SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
# SBATCH --gres=gpu:1                # number of GPUs (1/4 of GPUs)
# SBATCH --cpus-per-task=20           # number of cores per task (1/4 of the 4-GPUs node)
## SBATCH --hint=nomultithread
# SBATCH --qos=qos_gpu-t3
# SBATCH --time=00:20:00              # maximum execution time requested (HH:MM:SS)

cd ${SLURM_SUBMIT_DIR}

module purge
module load anaconda-py3/
source /gpfslocalsup/pub/anaconda-py3/2021.05/etc/profile.d/conda.sh

conda activate sb 

make clean haste fast_ligru

pip3 install fast_ligru-*.whl --force-reinstall

python3 /gpfswork/rech/vfw/ujg45iy/fast_ligru/frameworks/pytorch/ligru_2_0.py
