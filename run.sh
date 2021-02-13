#!/bin/bash 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 180G
#SBATCH --time 20:00:00
#SBATCH --job-name=mode_1
#SBATCH --gres gpu:2


module load gcc/8.4.0-cuda  python/3.7.7
source /work/vita/sadegh/argo/argo-env/bin/activate
python -V
python train.py --config-module configs.deepsvg.hierarchical_ordered --data-type argo --modes 1 --obs_len 20  --pred_len 30  --normalize   --train_features /work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_train.pkl   --val_features /work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_val.pkl --test_features /work/vita/sadegh/argo/argoverse-forecasting/forecasting_features_test.pkl   
