# modified from :
# https://huggingface.co/Qwen/Qwen3-4B#quickstart
# https://stackoverflow.com/a/58829816
# https://github.com/huggingface/transformers/issues/10704#issuecomment-798870853

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import pickle
from extract_movies import extract_movies
from tqdm import tqdm


model_name = "Qwen/Qwen3-4B"
batch_size = 16

# Set padding left for decoder-only model
# Use padding to do batch processing, so that each input is in the fixed length.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"


# use 2 GPUs
model0 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:1")
model0.eval()
model1.eval()


# apply Qwen 3's chat template to conversation and prompt defined in paper 
def build_template(context, pred, candidates):

    movies = ""
    for idx, each in enumerate(pred):
        movies = movies + f"\n{idx}. {candidates[each]}"

    prompt = f"""Pretend you are an reranker in the movie recommender system. I will give you a conversation between a user and a movie recommender, along with a list of 20 recommended movies.

Based on the conversation, please rerank the list of recommended movies from most aligned with user's  preferences to least aligned. If you think the order of the given list of recommended movies already ranked from most aligned with the user's preferences to least aligned, do not reorder. Please reply me with an ordered movie list without extra sentences.

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


# send 8 test data points to model, then get model's movie recommendation for each data points
def generate_batch(batch_context, batch_pred, candidates, model, device):
    rec_movies_batch = []
    templated_texts = []
    
    for context, pred in zip(batch_context, batch_pred):
        templated_texts.append(build_template(context, pred, candidates))

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

    for text in decoded:
        rec_movies_batch.append(extract_movies(text, candidates))

    return rec_movies_batch


# worker thread
def worker(contexts_slice, context_pred, candidates, model, device):
    output = []

    for i in tqdm(range(0, len(contexts_slice), batch_size)):

        batch_context = contexts_slice[i : i + batch_size]
        batch_pred = context_pred[i : i + batch_size]

        output.extend(generate_batch(batch_context, batch_pred, candidates, model, device))

    return output


def main():
    datasets = {
        "reddit": './pred_result_reddit.pickle',
        "redial": './pred_result_redial.pickle',
        "inspired": './pred_result_inspired.pickle',
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

        # split testing data for two GPUs
        all_contexts = result_data["context"]
        all_pred = result_data["pred_label"]
        
        reordered_preds = []

        half = (len(all_contexts) + 1) // 2
        part1, part2 = all_contexts[:half], all_contexts[half:]
        part1_pred, part2_pred = all_pred[:half], all_pred[half:]

        # create two thread, each thread manage one GPU.
        with ThreadPoolExecutor(max_workers=2) as exe:
            f1 = exe.submit(worker, part1, part1_pred, result_data["candidates"], model0, "cuda:0")
            f2 = exe.submit(worker, part2, part2_pred, result_data["candidates"], model1, "cuda:1")

            preds = f1.result() + f2.result()


        # In main thread, save model's output to pickle file for later evaluation
        result_data["pred_label"] = preds
        with open(f"pred_result_{each}_rag.pickle", "wb") as f:
            pickle.dump(result_data, f)


if __name__ == "__main__":
    main()