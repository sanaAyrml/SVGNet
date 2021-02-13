#!/bin/bash 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 180G
#SBATCH --time 4:00:00
#SBATCH --job-name=mode_6
#SBATCH --gres gpu:2
#SBATCH --account vita

module load gcc/8.4.0-cuda  python/3.7.7
source /work/vita/sadegh/argo/argo-env/bin/activate
python -V
python inference.py 