
### Overveiw
The original work is from [NBCRS: Neighborhood-based Collaborative Filtering for Conversational Recommendation (Recsys 24)](https://github.com/zhouhanxie/neighborhood-based-CF-for-CRS/tree/main). For our replication job, We extended the evaluation by testing NBCRS on an additional dataset (GoRecDial) and comparing its performance against three baseline LLMs: Qwen, Mistral, and retrieval-augmented generation (RAG) versions of both models. We also used Mean Reciprocal Rank (MRR) as an additional evaluation metric to provide a more comprehensive assessment of recommendation quality.


### Datasets

- Processed datasets (Inspired, Redial, Reddit) are in ```datasets``` folder.
- We post-processed the additional dataset GoRecDial from [Recommendation as a Communication Game:
Self-Supervised Bot-Play for Goal-oriented Dialogue](https://arxiv.org/pdf/1909.03922), (can be found in [Google Drive](https://drive.google.com/drive/folders/1nilk6FUktW2VjNlATdM0VMehzSOPIvJ0)). The approach is mentioned below.

### Training

- Training code is in ```train_knnlm.py```, see ```modeling_nmf.py``` for the actual knnlm model.
- See ```train_knnlm.sh``` for bash commands for training the model.

### Inference
- see ```inference_knnlm.ipynb``` for the code for tuning number of neighbors to use for the KNN component and doing inference on test set for the datasets.

### Environment

- see ```requirements.txt``` which is exported via ```conda list -e > requirements.txt```.
- see ```requirenment_new1.txt``` if you want to build environment through ```pip install -r requirement_new1.txt```

### Procedure for running on a dataset:

- For convenience, we've already included fully processed datasets in the ```datasets/GoRecDial folder``` that are ready to use
- For those who wish to build custom train and test datasets, please download the raw files into the ```datasets/GoRecDial/raw_GoRecDial``` directory (you'll need to create this folder yourself). 
- To construct new train and test dataset, run ```GoRecDial_Post_processing.ipynb```. This file processes the raw files you downloaded in the previous step.

#### For NBCRS

- First enable your virutal environment, then install the environment via ```pip install -r requirements_new1.txt```.
- Then, execute the ```train.sh``` script to generate semantic embeddings through item-item co-occurrence matrix factorization and to train the models on each dataset. If you prefer to generate embeddings and train models separately, you can run these bash commands individually.
- Then Run ```inference_knnlm.ipynb``` to get evaluation results
   - Got way higher numbers for Inspired/Redial than in the paper? We found that the way we processed the data (including both movie and non-movie entities as target items during training while evaluating on predicting movies) results in lower numbers for the models compared to prior works.
   - Now, we exlucde non-movie entities during prediction by default. See usage of ```inspired/redial_eligible_entities``` variables for details in ```inference_knnlm.ipynb```; should be fairly easy to switch of this behavior by commenting out the post-filtering line.
- Run ```inference_knnlm_zero_shot.ipynb``` to get the zero-shot setting evaluation.

#### For LLMs
- There might be a conflict as LLMs are using newest dependency version. Recommend to create and activate another virtual enviroment first.
- run ```pip install -r requirements_new1.txt``` in either rag or qwen_only directory.

- In qwen_only folder:
   - For running Zero-shot Qwen3-4B to generate list of movies, run ```qwen3_4b_sequential.py```. If you are on multi-GPU platform, run ```qwen3_4b_parallel.py```. 
   - Run ```evaluation.py``` to get recall and MRR of Qwen.

- In rag folder:
   - For running Zero-shot Mistral to retrieve a list of movies, run mistral.py. We will have Mistral-only result after script finish executing
   - Then, we run Zero-shot Qwen3-4B to rerank the list of movies retrieved by Mistral. If you are on single-GPU platform, run ```qwen_reranker_sequential.py```. If you are on multi-GPU platform, run ```qwen_reranker_parallel.py```.
   - Finally, run ```evaluation.py``` to get recall and MRR of Mistral and Mistral+Qwen.

- The pickle file will be generated during the Qwen and Mistral inference, storing the predicted movie list for late evaluation. 
