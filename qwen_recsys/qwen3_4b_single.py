# If you are having a small size GPU, 
# this will be stater script for you to modify to speed up the inference

from transformers import AutoModelForCausalLM, AutoTokenizer
from extract_movies import extract_movies
from data_utils_for_llm import get_test_data
from tqdm import tqdm
import pickle


test_data, candidates = get_test_data()

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

output_data = {}

for input in tqdm(test_data["context"]):
    prompt = f"Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n{input}"

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    predicted_movie = extract_movies(content, candidates)

    test_data["pred_label"].append(predicted_movie)


with open("prediction_result.pickle", 'wb') as file:
    pickle.dump(test_data, file)