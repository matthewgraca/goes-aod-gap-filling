#!/bin/bash

#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N
#SBATCH --job-name=gap-filler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3090:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00

setenv SCRDIR /scratch/local/$USER/$SLURM_JOB_ID
setenv WORKDIR $HOME/output
cd $SCRDIR || exit

module load cuda/11.8.0
module use $HOME/MyModules
module load miniconda3/latest
source $HOME/software/pkg/miniconda3/etc/profile.d/conda.sh
conda activate research

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--standalone \
--nnodes=1 \
--nproc-per-node=4 \
/uufs/chpc.utah.edu/common/home/u6049013/aod-gapfiller/main.py
#--rdzv-id=$RANDOM \
#--rdzv-backend=c10d \
#--rdzv-endpoint=$head_node_ip:29500 \

cp -r $SCRDIR/* $WORKDIR/.