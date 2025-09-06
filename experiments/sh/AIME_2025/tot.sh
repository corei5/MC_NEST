#!/bin/zsh
#SBATCH --job-name=NEG_gpt4o
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=p_48G
#SBATCH --output=/nfs/home/rabbyg/CAG/log/AIME_2025/tot.log

export PYTHONPATH=/nfs/home/rabbyg/CAG/tree-of-thought-llm/src:$PYTHONPATH

sed -i 's/openai.error.OpenAIError/openai.OpenAIError/g' /nfs/home/rabbyg/CAG/tree-of-thought-llm/src/tot/models.py

source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/gpt4o/tot_aime.py
