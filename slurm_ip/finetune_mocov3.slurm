#!/bin/bash
#SBATCH --job-name=finetune_mocov3   # create a short name for your job 
#SBATCH --partition gpu-ee
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=2                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2               # number of gpus per node
#SBATCH --time=50:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=bdedhia@princeton.edu
#SBATCH -o ../slurm_op/mocov3_finetune.out
module purge
module load anaconda3/2020.7


conda activate txf_design-space
cd ..
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env finetune_mocov3.py --output_dir ./logs_dir/mocov3_finetune/ --input-size 224 --batch-size 256 --data-path ../ILSVRC2012/ --epochs 50 --dist-eval --resume ./model_weights/mocov3_s.pth