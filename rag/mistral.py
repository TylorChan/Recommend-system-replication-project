import pickle
from data_utils_for_rag import get_test_data
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np


# Load the model
model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")

# Each query must come with a one-sentence instruction that describes the task
task = f'Given a conversation between a user and a movie recommender. \nBased on the conversation, retrieve movies that are most suitable for the user.'

prompt = f"Instruct: {task}\nQuery: "


def main():
    datasets = {
        "reddit": '../datasets/reddit/reddit_test.csv',
        "redial": '../datasets/redial/redial_test.csv',
        "inspired": '../datasets/inspired/inspired_test.csv',
    }

    for each in datasets:
        preds = []
        passages = []
        scores = []
        print(f"Working on {each}")

        # test_data is a dictionary 
        # {"context": conversation, 
        # "label": movie id correspond to the conversation, 
        # "pred_label": predicted movies wait to be filled by model}

        # candidates is a list of all possible movies
        test_data, candidates = get_test_data(datasets[each])

        queries = test_data["context"]

        for candidate in candidates:
            passages.append(f"I would recommend {candidate}.")

        # encode movies
        passage_embeddings = model.encode(passages, batch_size=128, show_progress_bar=True)

        for query in tqdm(queries):
            # Encode the query
            query_embeddings = model.encode(query, prompt=prompt)
            
            # Compute the similarity scores
            scores = model.similarity(query_embeddings, passage_embeddings) * 100

            # get the top-2 for current query
            top_20_idx = np.argsort(scores.tolist()[0])[::-1][:20]
            preds.append(top_20_idx)

        # save model's output to pickle file for later evaluation
        test_data["pred_label"] = preds
        with open(f"pred_result_{each}.pickle", "wb") as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    main()