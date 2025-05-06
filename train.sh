python -u generate_embeddings.py \
    --input_file_dir 'datasets/inspired/inspired_train.csv' \
    --output_file_dir 'semantic_embs_inspired.pt'

python -u generate_embeddings.py \
    --input_file_dir 'datasets/redial/redial_train.csv' \
    --output_file_dir 'semantic_embs_redial.pt' 

python -u generate_embeddings.py \
    --input_file_dir 'datasets/reddit/reddit_large_train.csv' \
    --output_file_dir 'semantic_embs_reddit.pt' 