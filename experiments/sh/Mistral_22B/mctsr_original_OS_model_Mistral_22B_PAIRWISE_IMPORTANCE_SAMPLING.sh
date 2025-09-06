#!/bin/zsh
#SBATCH --job-name=mis_OP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:a3090:2
#SBATCH --partition=p_48G  # Specify the partition explicitly
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/Mistral_22B/mctsr_original_OS_model_Mistral_22B_PAIRWISE_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/Mistral_22B/mctsr_original_OS_model_Mistral_22B_PAIRWISE_IMPORTANCE_SAMPLING.py