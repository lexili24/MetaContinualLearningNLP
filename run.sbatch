#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=160000
#SBATCH --time=0:60:00
#SBATCH --job-name="nlu_project"

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=dj1057
source activate /scratch/dj1057/nlu/env
export PYTORCH_TRANSFORMERS_CACHE=/scratch/${NETID}/nlu/cache
conda install pytorch torchvision -c pytorch

cd /home/dj1057/NLU
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH

python main.py
