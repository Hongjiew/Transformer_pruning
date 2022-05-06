#!/bin/bash
#SBATCH --job-name=random_deit_base    # create a short name for your job 
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1               # number of gpus per node
#SBATCH --time=00:10:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=hw4948@princeton.edu
#SBATCH -o ../slurm_op/random_deit_base.out

module purge
module load anaconda3/2021.5


conda activate tokenrank
cd ..
python3 zero_shot_pruning_eval.py --model_name='tokenrank_deit_base_3' --prune_list 2 5 8 --retain_rate_list 0.9 0.8 0.7 --data-path='/scratch/gpfs/hw4948/ILSVRC2012'