# modified from :
# https://huggingface.co/Qwen/Qwen3-4B#quickstart
# https://stackoverflow.com/a/58829816
# https://github.com/huggingface/transformers/issues/10704#issuecomment-798870853

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import pickle
from extract_movies_for_rag import extract_movies
from tqdm import tqdm


model_name = "Qwen/Qwen3-4B"
batch_size = 16

# Set padding left for decoder-only model
# Use padding to do batch processing, so that each input is in the fixed length.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"


# use 2 GPUs
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
model.eval()


# apply Qwen 3's chat template to conversation and prompt defined in paper 
def build_template(context, pred, candidates):

    movies = ""
    for idx, each in enumerate(pred):
        movies = movies + f"\n{idx}. {candidates[each]}"

    prompt = f"""Pretend you are an reranker in the movie recommender system. I will give you a conversation between a user and a movie recommender, along with a list of 20 recommended movies.

Based on the conversation, please rerank the list of recommended movies from most aligned with user's  preferences to least aligned. If you think the order of the given list of recommended movies already ranked from most aligned with the user's preferences to least aligned, do not reorder. Please reply me with an ordered list of 20 movies without extra sentences.

Here is the conversation:
{context}

Here is the list of recommended movies:{movies}
"""
    
    messages = [{"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


# send batch_size test data points to model, then get model's movie recommendation for each data points
def generate_batch(batch_context, batch_pred, candidates, model, device):
    rec_movies_batch = []
    templated_texts = []

    movie2id = {each: idx for idx, each in enumerate(candidates)}
    pred_candidates = []

    for context, pred in zip(batch_context, batch_pred):
        templated_texts.append(build_template(context, pred, candidates))
        pred_candidates.append([candidates[each] for each in pred])

    inputs = tokenizer(
        templated_texts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7, 
            top_p=0.8, 
            top_k=20,
            min_p=0
        )

    output_token_ids = []
    for idx, seq in enumerate(outputs):
        output_token_ids.append(seq[len(inputs[idx]) : ].tolist())

    decoded = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)

    for idx, text in enumerate(decoded):
        rec_movies_batch.append(extract_movies(text, pred_candidates[idx], movie2id))

    return rec_movies_batch


def main():
    datasets = {
        "reddit": './pred_result_reddit.pickle',
        "redial": './pred_result_redial.pickle',
        "inspired": './pred_result_inspired.pickle',
        "GoRecDial": './pred_result_GoRecDial.pickle'
    }

    for each in datasets:
        print(f"Working on {each}")

        # result_data is a dictionary 
        # {"context": conversation, 
        # "label": movie id correspond to the conversation, 
        # "candidates": all possible movies
        # "pred_label": predicted movies filled by text embedding model}
        with open(f"pred_result_{each}.pickle", "rb") as file:
            result_data = pickle.load(file)

        all_contexts = result_data["context"]
        all_pred = result_data["pred_label"]

        reordered_preds = []

        for i in tqdm(range(0, len(all_contexts), batch_size)):

            batch_context = all_contexts[i : i + batch_size]
            batch_pred = all_pred[i : i + batch_size]
            batch_output = generate_batch(batch_context, batch_pred, result_data["candidates"], model, "cuda:0")
            
            reordered_preds.extend(batch_output)

        # In main thread, save model's output to pickle file for later evaluation
        test_data["pred_label"] = reordered_preds
        with open(f"pred_result_{each}.pickle", "wb") as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    main()