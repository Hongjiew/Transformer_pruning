#!/bin/bash
#SBATCH --job-name=deit_small_distill_987_em_n     # create a short name for your job 
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2               # number of gpus per node
#SBATCH --time=80:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=hw4948@princeton.edu
#SBATCH -o ../slurm_op/deit_small_epoch50_987_em_n.out

module purge
module load anaconda3/2021.5


conda activate tokenrank
cd ..
python3 -u -m torch.distributed.launch --nproc_per_node=2 --use_env main_em.py  --output_dir ./logs_dir/deit_small_distill_987_em_n/ --model_name tokenrank_deit_small_3 --input-size 224 --batch-size 256 --data-path ../ILSVRC2012/ --epochs 80 --dist-eval --retain_rate_list 0.9 0.8 0.7 --distill_model --lr=1e-5 --min-lr=5e-6
