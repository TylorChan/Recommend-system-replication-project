# modified from :
# https://huggingface.co/Qwen/Qwen3-4B#quickstart
# https://stackoverflow.com/a/58829816
# https://github.com/huggingface/transformers/issues/10704#issuecomment-798870853

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import pickle
from extract_movies import extract_movies
from data_utils_for_llm import get_test_data
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
def build_template(context):
    prompt = f"Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system). \nBased on the conversation, you reply me with 20 recommendations without extra sentences. \nHere is the conversation: {context}"
    
    messages = [{"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


# send batch_size test data points to model, then get model's movie recommendation for each data points
def generate_batch(model, batch_context, candidates, device):
    rec_movies_batch = []
    templated_texts = []
    
    for context in batch_context:
        templated_texts.append(build_template(context))

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


def main():
    datasets = {
            "reddit": '../datasets/reddit/reddit_test.csv',
            "redial": '../datasets/redial/redial_test.csv',
            "inspired": '../datasets/inspired/inspired_test.csv',
            "GoRecDial": '../datasets/GoRecDial/GoRecDial_test.csv'
        }

    for each in datasets:
        print(f"Working on {each}")
        # test_data is a dictionary 
        # {"context": conversation, 
        # "label": movie id correspond to the conversation, 
        # "pred_label": predicted movies wait to be filled by model}

        # candidates is a list of all possible movies
        test_data, candidates = get_test_data(datasets[each])

        # split testing data for two GPUs
        all_contexts = test_data["context"]
        preds = []

        for i in tqdm(range(0, len(all_contexts), batch_size)):

            batch_context = all_contexts[i : i + batch_size]

            preds.extend(generate_batch(model, batch_context, candidates, "cuda:0"))


        # In main thread, save model's output to pickle file for later evaluation
        test_data["pred_label"] = preds
        with open(f"pred_result_{each}.pickle", "wb") as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    main()