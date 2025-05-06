#!/usr/bin/env python
# coding: utf-8

# In[2]:


from modeling_nmf import NMF, NMFConfig
from data_utils import get_reddit_data
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# In[3]:


from tqdm import tqdm

def build_datastore_embedding(sentences, model, tokenizer, device):
    print('building datastore embeddings')
    model.eval()
    
    # Initialize an empty list to hold the sentence embeddings
    all_sentence_embeddings = []
    
    # Process sentences in batches
    batch_size = 64
    
    for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences)//batch_size):
        batch_sentences = sentences[i:i+batch_size]
        
        # Tokenize sentences
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, output_hidden_states=True)
    
        # # Perform pooling
        # sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
    
        # insteading of doing pooling, can just use last hidden state
        sentence_embeddings = model_output.hidden_states[-1][:,0,:]
        
        all_sentence_embeddings.append(sentence_embeddings.cpu())
    
    # Concatenate all batched embeddings
    all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)
    
    all_sentence_embeddings = F.normalize(all_sentence_embeddings, p=2, dim=1).cpu()
    post_embeddings = all_sentence_embeddings

    return post_embeddings


# In[4]:


class KNNLMForTuningHyperParams:

    def __init__(
        self, 
        data_path,
        tokenizer_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
        model_name_or_path = 'saved_models'
    ):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentence_embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        nmf = NMF.from_pretrained(model_name_or_path).to(device)
        nmf.eval()
        sentence_embedding_model = nmf.model
        
        training_dataset, validation_dataset, movie_vocab = get_reddit_data_with_heldout(data_path, heldout_portion=0.2)
        movie_vocab = [m.split(' (')[0] for m in movie_vocab] # remove the year at end
        training_posts = sorted(list(set(training_dataset['context'])))
        trainingpost2idx = dict(zip(
            training_posts, 
            list(range(len(training_posts)))
        ))

        trainingpostidx2movies = defaultdict(list)
        movie_from_posts = []
        for i in range(len(training_dataset['context'])):
            post_idx = trainingpost2idx[training_dataset['context'][i]]
            # the split is for removing year at end, e.g. " (2019)"
            # movie = movie_vocab[training_dataset['label'][i]].split(' (')[0] 
            trainingpostidx2movies[post_idx].append(training_dataset['label'][i])

        post_embeddings = build_datastore_embedding(
            sentences = training_posts, 
            model = sentence_embedding_model, 
            tokenizer = sentence_embedding_tokenizer, 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).to(device)

        self.device = device
        self.sentence_embedding_tokenizer = sentence_embedding_tokenizer
        self.sentence_embedding_model = sentence_embedding_model
        self.movie_vocab = movie_vocab
        self.training_posts = training_posts
        self.post_embeddings = post_embeddings
        self.trainingpostidx2movies = trainingpostidx2movies
        self.nmf = nmf
        self.validation_dataset = validation_dataset

    def encode_sentences(self, batch_sentences):
        with torch.no_grad():
            encoded_input = self.sentence_embedding_tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                model_output = self.sentence_embedding_model(**encoded_input, output_hidden_states=True)
            sentence_embeddings = model_output.hidden_states[-1][:,0,:]
    
        return sentence_embeddings

    def top_post_ids_retrieval(self, query):
    
        query_embedding = self.encode_sentences(query)[0].reshape(1, -1)
        cosine_similarities = torch.cosine_similarity(query_embedding, self.post_embeddings).cpu().numpy()
        sorted_indices = np.argsort(cosine_similarities)[::-1]
        return sorted_indices, cosine_similarities[sorted_indices]

    def count_based_probability(
        self,
        query, 
        num_posts_to_consider=30, 
        return_logits=False, 
        distance_weighting=False,
        temperature = 1.0
    ):
        relevant_post_ids, similarities = self.top_post_ids_retrieval(query)
        movie_pool = defaultdict(int)
        probas = np.zeros(len(self.movie_vocab))
        for i, id in enumerate(relevant_post_ids[:num_posts_to_consider]):
            for movieid in self.trainingpostidx2movies[id]:
                if distance_weighting:
                    probas[movieid] += similarities[i]/temperature
                else:
                    probas[movieid] += 1
        if return_logits:
            return probas
        probas = torch.nn.functional.softmax(torch.from_numpy(probas), dim=-1)
        return probas

    def predictor_probability(
        self, 
        query,
        return_logits = False
    ):
        
        with torch.no_grad():
            model_input = self.sentence_embedding_tokenizer(
                query, 
                return_tensors='pt', 
                max_length=368, 
                truncation=True
            )

            logits = reddit_knnlm_recommender.nmf(
                            model_input['input_ids'].to(self.device), 
                            model_input['token_type_ids'].to(self.device), 
                            model_input['attention_mask'].to(self.device),
                            labels=None
                        ).logits
        if return_logits:
            return logits.cpu().numpy()
        probas = F.softmax(logits, dim=-1)[0]
        return  probas.cpu().numpy()


# In[5]:


class KNNLMRecommender:

    def __init__(
        self, 
        tokenizer_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2',
        model_name_or_path = 'saved_models',
        data_path = 'reddit/reddit_large_train.csv'
    ):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentence_embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        nmf = NMF.from_pretrained(model_name_or_path).to(device)
        nmf.eval()
        sentence_embedding_model = nmf.model
        
        training_dataset, movie_vocab = get_reddit_data(data_path)
        movie_vocab = [m.split(' (')[0] for m in movie_vocab] # remove the year at end
        training_posts = sorted(list(set(training_dataset['context'])))
        trainingpost2idx = dict(zip(
            training_posts, 
            list(range(len(training_posts)))
        ))

        trainingpostidx2movies = defaultdict(list)
        movie_from_posts = []
        for i in range(len(training_dataset['context'])):
            post_idx = trainingpost2idx[training_dataset['context'][i]]
            # the split is for removing year at end, e.g. " (2019)"
            # movie = movie_vocab[training_dataset['label'][i]].split(' (')[0] 
            trainingpostidx2movies[post_idx].append(training_dataset['label'][i])

        post_embeddings = build_datastore_embedding(
            sentences = training_posts, 
            model = sentence_embedding_model, 
            tokenizer = sentence_embedding_tokenizer, 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.device = device
        self.sentence_embedding_tokenizer = sentence_embedding_tokenizer
        self.sentence_embedding_model = sentence_embedding_model
        self.movie_vocab = movie_vocab
        self.training_posts = training_posts
        self.post_embeddings = post_embeddings
        self.trainingpostidx2movies = trainingpostidx2movies
        self.nmf = nmf

    def encode_sentences(self, batch_sentences):
        with torch.no_grad():
            encoded_input = self.sentence_embedding_tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                model_output = self.sentence_embedding_model(**encoded_input, output_hidden_states=True)
            sentence_embeddings = model_output.hidden_states[-1][:,0,:]
    
        return sentence_embeddings

    def top_post_ids_retrieval(self, query):
    
        query_embedding = self.encode_sentences(query)[0].reshape(1, -1).cpu()
        cosine_similarities = cosine_similarity(query_embedding, self.post_embeddings)
        sorted_indices = np.argsort(cosine_similarities[0])[::-1]
        return sorted_indices, cosine_similarities[0][sorted_indices]

    def count_based_probability(
        self,
        query, 
        num_posts_to_consider=30, 
        return_logits=False, 
        distance_weighting=False
    ):
        relevant_post_ids, similarities = self.top_post_ids_retrieval(query)
        movie_pool = defaultdict(int)
        probas = np.zeros(len(self.movie_vocab))
        for i, id in enumerate(relevant_post_ids[:num_posts_to_consider]):
            for movieid in self.trainingpostidx2movies[id]:
                if distance_weighting:
                    probas[movieid] += similarities[i]
                else:
                    probas[movieid] += 1
        if return_logits:
            return probas
        probas = torch.nn.functional.softmax(torch.from_numpy(probas), dim=-1)
        return probas

    def predictor_probability(
        self, 
        query,
        return_logits = False
    ):
        
        with torch.no_grad():
            model_input = self.sentence_embedding_tokenizer(
                query, 
                return_tensors='pt', 
                max_length=368, 
                truncation=True
            )

            logits = reddit_knnlm_recommender.nmf(
                            model_input['input_ids'].to(self.device), 
                            model_input['token_type_ids'].to(self.device), 
                            model_input['attention_mask'].to(self.device),
                            labels=None
                        ).logits
        if return_logits:
            return logits.cpu().numpy()
        probas = F.softmax(logits, dim=-1)[0]
        return  probas.cpu().numpy()


# In[6]:


import json

def get_eligible_entities(resource_path=''):
    entity2id = eval(open(resource_path+'entity2id.json', 'r').readlines()[0])
    id2entity = {v:k for k,v in entity2id.items()}
    eligible_entities = [id2entity[idx].split('/')[-1].split('_(')[0].rstrip('>').replace('_',' ') for idx in \
    eval(open(resource_path+'item_ids.json', 'r').readlines()[0])]
    return eligible_entities


# In[7]:


inspired_eligible_entities = set(get_eligible_entities('./entity_assets/inspired/'))


# In[8]:


redial_eligible_entities = set(get_eligible_entities('./entity_assets/redial/'))


# ## Inspired

# In[ ]:


from data_utils import get_reddit_data_with_heldout
import numpy as np
import pandas as pd
from scipy.stats import sem

reddit_knnlm_recommender = KNNLMForTuningHyperParams(
    data_path= 'datasets/inspired/inspired_train.csv',
    model_name_or_path = 'models/inspired'
)

k=20
n_neighbors = [15, 30, 60, 90, 120, 150, 180]
recall = []
for num_posts_to_consider in n_neighbors:
    hits_at_k = []
    mrr = []
    cache = dict()
    for idx in tqdm(range(len(reddit_knnlm_recommender.validation_dataset['context'])), total=len(reddit_knnlm_recommender.validation_dataset['context'])):
        query = reddit_knnlm_recommender.validation_dataset['context'][idx]
        target = reddit_knnlm_recommender.movie_vocab[reddit_knnlm_recommender.validation_dataset['label'][idx]]
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=num_posts_to_consider, 
                return_logits=True
            )
            print(f'This is movie_counts: {movie_counts}')
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            recommended_movies = [e for e in recommended_movies if e in inspired_eligible_entities]
        else:
            recommended_movies = cache[query]
        
        hits_at_k.append(int(target in recommended_movies[:k]))
        
    recall.append(np.mean(hits_at_k))


best_n_neighbors = n_neighbors[np.argmax(recall)]
print('the recommended number of neighbors to use is ', best_n_neighbors)

testset = pd.read_csv('datasets/inspired/inspired_test.csv')
test_inputs = testset['test_inputs']
test_groundtruths = testset['test_outputs']

reddit_knnlm_recommender = KNNLMRecommender(
    data_path= 'datasets/inspired/inspired_train.csv',
    model_name_or_path = 'models/inspired'
)

K = [1,5, 10, 20,50,100,300]

print('######################################################################')
print('######################## inspired ######################################')
print('######################################################################\n')

print('-------------------- Retrieval -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=True
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            recommended_movies = [e for e in recommended_movies if e in inspired_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))

print('-------------------- Recommend -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_probas)]
            recommended_movies = [e for e in recommended_movies if e in inspired_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# print('-------------------- R+R (rerank) -------------------------------')

# cache = dict()
# for k in K:
#     hits_at_k = []
#     for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
#         query = test_inputs[idx]
#         target = test_groundtruths[idx]
        
#         if query not in cache:
#             movie_counts = reddit_knnlm_recommender.count_based_probability(
#                 query, 
#                 num_posts_to_consider=best_n_neighbors, 
#                 return_logits=True
#             )
#             movie_probas = reddit_knnlm_recommender.predictor_probability(
#                 query, 
#                 return_logits=False
#             )
#             movie_scores = movie_counts + movie_probas
#             recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
#             recommended_movies = [e for e in recommended_movies if e in inspired_eligible_entities]
#             cache[query] = recommended_movies
#         else:
#             recommended_movies = cache[query]
#         hits_at_k.append(int(target in recommended_movies[:k]))
#     print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))

print('-------------------- R+R (rerank) with small gamma (cleaner solution for paper) -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=False
            )
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            movie_scores = movie_counts*(1-(1e-10)) + movie_probas*1e-10
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
            recommended_movies = [e for e in recommended_movies if e in inspired_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# ## Reddit

# In[ ]:


from data_utils import get_reddit_data_with_heldout
import numpy as np
import pandas as pd
from scipy.stats import sem

reddit_knnlm_recommender = KNNLMForTuningHyperParams(
    data_path= 'datasets/reddit/reddit_large_train.csv',
    model_name_or_path = 'models/reddit'
)

k=20
n_neighbors = [15, 30, 60, 90, 120, 150, 180]
recall = []
for num_posts_to_consider in n_neighbors:
    hits_at_k = []
    cache = dict()
    for idx in tqdm(range(len(reddit_knnlm_recommender.validation_dataset['context'])), total=len(reddit_knnlm_recommender.validation_dataset['context'])):
        query = reddit_knnlm_recommender.validation_dataset['context'][idx]
        target = reddit_knnlm_recommender.movie_vocab[reddit_knnlm_recommender.validation_dataset['label'][idx]]
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=num_posts_to_consider, 
                return_logits=True
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
        else:
            recommended_movies = cache[query]
        
        hits_at_k.append(int(target in recommended_movies[:k]))
    recall.append(np.mean(hits_at_k))


best_n_neighbors = n_neighbors[np.argmax(recall)]
print('the recommended number of neighbors to use is ', best_n_neighbors)

# best_n_neighbors = 30 # uncomment above to tune n_neighbors

testset = pd.read_csv('datasets/reddit/reddit_test.csv')
test_inputs = testset['test_inputs']
test_groundtruths = testset['test_outputs']

reddit_knnlm_recommender = KNNLMRecommender(
    data_path= 'datasets/reddit/reddit_large_train.csv',
    model_name_or_path = 'models/reddit'
)

K = [1,5, 10, 20,50,100,300]
print('######################################################################')
print('######################## reddit ######################################')
print('######################################################################\n')
print('-------------------- Retrieval -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=True
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))

print('-------------------- Recommend -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_probas)]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# print('-------------------- R+R (rerank) -------------------------------')

# cache = dict()
# for k in K:
#     hits_at_k = []
#     for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
#         query = test_inputs[idx]
#         target = test_groundtruths[idx]
        
#         if query not in cache:
#             movie_counts = reddit_knnlm_recommender.count_based_probability(
#                 query, 
#                 num_posts_to_consider=best_n_neighbors, 
#                 return_logits=True
#             )
#             movie_probas = reddit_knnlm_recommender.predictor_probability(
#                 query, 
#                 return_logits=False
#             )
#             movie_scores = movie_counts + movie_probas
#             recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
#             cache[query] = recommended_movies
#         else:
#             recommended_movies = cache[query]
#         hits_at_k.append(int(target in recommended_movies[:k]))
#     print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))

print('-------------------- R+R (rerank) with small gamma (cleaner solution for paper) -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=False
            )
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            movie_scores = movie_counts*(1-(1e-10)) + movie_probas*1e-10
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# ## Redial

# In[15]:


from data_utils import get_reddit_data_with_heldout
import numpy as np
import pandas as pd
from scipy.stats import sem

reddit_knnlm_recommender = KNNLMForTuningHyperParams(
    data_path= 'datasets/redial/redial_train.csv',
    model_name_or_path = 'models/redial'
)

k=20
n_neighbors = [15, 30, 60, 90, 120, 150, 180]
recall = []
for num_posts_to_consider in n_neighbors:
    hits_at_k = []
    cache = dict()
    for idx in tqdm(range(len(reddit_knnlm_recommender.validation_dataset['context'])), total=len(reddit_knnlm_recommender.validation_dataset['context'])):
        query = reddit_knnlm_recommender.validation_dataset['context'][idx]
        target = reddit_knnlm_recommender.movie_vocab[reddit_knnlm_recommender.validation_dataset['label'][idx]]
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=num_posts_to_consider, 
                return_logits=True
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            recommended_movies = [e for e in recommended_movies if e in redial_eligible_entities]
        else:
            recommended_movies = cache[query]
        
        hits_at_k.append(int(target in recommended_movies[:k]))
    recall.append(np.mean(hits_at_k))


best_n_neighbors = n_neighbors[np.argmax(recall)]
print('the recommended number of neighbors to use is ', best_n_neighbors)

# best_n_neighbors = 60

testset = pd.read_csv('datasets/redial/redial_test.csv')
test_inputs = testset['test_inputs']
test_groundtruths = testset['test_outputs']

reddit_knnlm_recommender = KNNLMRecommender(
    data_path= 'datasets/redial/redial_train.csv',
    model_name_or_path = 'models/redial'
)

K = [1,5, 10, 20,50,100,300]

print('######################################################################')
print('######################## Redial ######################################')
print('######################################################################\n')


print('-------------------- Retrieval -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=True
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            recommended_movies = [e for e in recommended_movies if e in redial_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))

print('-------------------- Recommend -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_probas)]
            recommended_movies = [e for e in recommended_movies if e in redial_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# print('-------------------- R+R (rerank) -------------------------------')

# cache = dict()
# for k in K:
#     hits_at_k = []
#     for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
#         query = test_inputs[idx]
#         target = test_groundtruths[idx]
        
#         if query not in cache:
#             movie_counts = reddit_knnlm_recommender.count_based_probability(
#                 query, 
#                 num_posts_to_consider=best_n_neighbors, 
#                 return_logits=True
#             )
#             movie_probas = reddit_knnlm_recommender.predictor_probability(
#                 query, 
#                 return_logits=False
#             )
#             movie_scores = movie_counts + movie_probas
#             recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
#             recommended_movies = [e for e in recommended_movies if e in redial_eligible_entities]
#             cache[query] = recommended_movies
#         else:
#             recommended_movies = cache[query]
#         hits_at_k.append(int(target in recommended_movies[:k]))
#     print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


print('-------------------- R+R (rerank) with small gamma (cleaner solution for paper) -------------------------------')

cache = dict()
for k in K:
    hits_at_k = []
    for idx in tqdm(range(len(test_inputs)), total = len(test_inputs) ):
        query = test_inputs[idx]
        target = test_groundtruths[idx]
        
        if query not in cache:
            movie_counts = reddit_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=best_n_neighbors, 
                return_logits=False
            )
            movie_probas = reddit_knnlm_recommender.predictor_probability(
                query, 
                return_logits=False
            )
            movie_scores = movie_counts*(1-(1e-10)) + movie_probas*1e-10
            recommended_movies = np.array(reddit_knnlm_recommender.movie_vocab)[np.argsort(-movie_scores)]
            recommended_movies = [e for e in recommended_movies if e in redial_eligible_entities]
            cache[query] = recommended_movies
        else:
            recommended_movies = cache[query]
        hits_at_k.append(int(target in recommended_movies[:k]))
    print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))


# In[ ]:




