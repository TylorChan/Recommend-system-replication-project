
### Overveiw
The original work is from [NBCRS: Neighborhood-based Collaborative Filtering for Conversational Recommendation (Recsys 24)](https://github.com/zhouhanxie/neighborhood-based-CF-for-CRS/tree/main). Our replication project is to test NBCRS with an additional baseline model, conversatinal datasets and metric (NDCG).

### Datasets

- Processed datasets (Inspired, Redial, Reddit) are in ```datasets``` folder.
- The additional dataset we want to test is from [Recommendation as a Communication Game:
Self-Supervised Bot-Play for Goal-oriented Dialogue](https://arxiv.org/pdf/1909.03922), (can be found in [Google Drive](https://drive.google.com/drive/folders/1nilk6FUktW2VjNlATdM0VMehzSOPIvJ0))
- 

### Training

- Training code is in ```train_knnlm.py```, see ```modeling_nmf.py``` for the actual knnlm model.
- See ```train_knnlm.sh``` for bash commands for training the model.

### Inference
- see ```inference_knnlm.ipynb``` for the code for tuning number of neighbors to use for the KNN component and doing inference on test set for the datasets.

### Environment

- see ```requirements.txt``` which is exported via ```conda list -e > requirements.txt```.
- see ```requirenment_new.txt``` if you want to build environment through ```pip install -r requirement_new.txt```

### Procedure for running on a dataset:

- First, run ```generate_embeddings.py```, to generate semantic embeddings by factorizing item-item co-occur matrix. (mainly to stablize training)
- Then, run ```train_knnlm.py```. We use huggingface style training pipeline.
- Run ```inference_knnlm.ipynb``` to get evaluation results
   - Got way higher numbers for Inspired/Redial than in the paper? We found that the way we processed the data (including both movie and non-movie entities as target items during training while evaluating on predicting movies) results in lower numbers for the models compared to prior works.
   - Now, we exlucde non-movie entities during prediction by default. See usage of ```inspired/redial_eligible_entities``` variables for details in ```inference_knnlm.ipynb```; should be fairly easy to switch of this behavior by commenting out the post-filtering line.
   



