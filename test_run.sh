#!/bin/bash
#
#SBATCH -p gpu
#SBATCH --job-name=test_run
#SBATCH --output=test_run.txt
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=1-23:59:59
##SBATCH --gpus 2
#SBATCH --cpus-per-gpu=10
##SBATCH --mem=256000
#SBATCH --mem-per-gpu=24G

echo "start!"
source activate /home/groups/quake/jingyi_wei/miniconda3/envs/casrx
cd /oak/stanford/groups/silvanak/jingyi/casrx_guide_efficiency/CasRx_guide_efficiency/models/Deep-learning

for s in {0..8}
do
python3 train.py --dataset guide_all_features_9f --model guide_nolin_ninef --kfold 9 --split $s
done


echo "done!"
