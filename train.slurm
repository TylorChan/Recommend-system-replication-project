#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=40g
#SBATCH --tmp=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7751@umn.edu
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:a40:1


export PATH=/users/9/chen7751/miniconda3/envs/5123/bin:$PATH

python -u generate_embeddings.py \
    --input_file_dir 'datasets/inspired/inspired_train.csv' \
    --output_file_dir 'semantic_embs_inspired.pt'

python -u train_knnlm.py \
    --pretrained_embedding_path 'semantic_embs_inspired.pt' \
    --n_item_latent_factors 16 \
    --output_dir './models/inspired' \
    --training_data_path 'datasets/inspired/inspired_train.csv' \
    --max_movies 20000

python -u generate_embeddings.py \
    --input_file_dir 'datasets/redial/redial_train.csv' \
    --output_file_dir 'semantic_embs_redial.pt' 

python -u train_knnlm.py \
    --pretrained_embedding_path 'semantic_embs_redial.pt' \
    --n_item_latent_factors 16 \
    --output_dir './models/redial' \
    --training_data_path 'datasets/redial/redial_train.csv' \
    --max_movies 20000

python -u generate_embeddings.py \
    --input_file_dir 'datasets/reddit/reddit_large_train.csv' \
    --output_file_dir 'semantic_embs_reddit.pt' 

python -u train_knnlm.py \
    --pretrained_embedding_path 'semantic_embs_reddit.pt' \
    --n_item_latent_factors 16 \
    --output_dir './models/reddit' \
    --training_data_path 'datasets/reddit/reddit_large_train.csv' \
    --max_movies 20000

# jupyter nbconvert --to python inference_knnlm.ipynb

# python inference_knnlm.py