import pandas as pd

def get_test_data(csv_path='../datasets/reddit/reddit_test.csv'):
   
    test_df = pd.read_csv(csv_path)
    candidates = test_df["test_outputs"].unique()
    movie2id = {each: idx for idx, each in enumerate(candidates)}
    
    test_output_ids = []
    for each in test_df["test_outputs"]:
        test_output_ids.append(movie2id[each])

    test_dict = {
        "context": test_df["test_inputs"],
        "label": test_output_ids,
        "candidates": candidates,
        "pred_label": []
    }

    return test_dict, candidates
