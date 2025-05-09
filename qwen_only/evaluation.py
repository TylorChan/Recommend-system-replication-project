import pickle
import numpy as np

for dataset in ["reddit", "inspired", "redial", "GoRecDial"]:
    print(dataset)
    test_data ={}
    mrr5 = []
    mrr10 = []
    mrr20 = []

    with open(f"pred_result_{dataset}.pickle", "rb") as file:
        test_data = pickle.load(file)

    count1 = 0
    count5 = 0
    count20 = 0

    for idx, each in enumerate(test_data["label"]):
        # calculate Recall
        if each == test_data["pred_label"][idx][0]:
            count1 += 1
        if each in test_data["pred_label"][idx][:5]:
            count5 += 1
        if each in test_data["pred_label"][idx][:20]:
            count20 += 1
        
        # calculate MRR
        target_rank = 0
        if each in test_data["pred_label"][idx][:20]:
            target_index =  list(test_data["pred_label"][idx])[:20].index(each)
            target_rank = 1/(target_index+1)
            mrr20.append(target_rank)
        else:
            mrr20.append(0)
            
        if each in test_data["pred_label"][idx][:10]:
            target_index =  list(test_data["pred_label"][idx])[:10].index(each)
            target_rank = 1/(target_index+1)
            mrr10.append(target_rank)
        else:
            mrr10.append(0)

        if each in test_data["pred_label"][idx][:5]:
            target_index =  list(test_data["pred_label"][idx])[:5].index(each)
            target_rank = 1/(target_index+1)
            mrr5.append(target_rank)
        else:
            mrr5.append(0)

    
    hit1 = (count1/ len(test_data["pred_label"])) * 100
    hit5 = (count5/ len(test_data["pred_label"])) * 100
    hit20 = (count20/ len(test_data["pred_label"])) * 100

    print("hit@1:", f"%{hit1}")
    print("hit@5:", f"%{hit5}")
    print("hit@20:", f"%{hit20}")
    print("MRR@5:", f"{np.mean(mrr5) * 100}")
    print("MRR@10:", f"{np.mean(mrr10) * 100}")
    print("MRR@20:", f"{np.mean(mrr20) * 100}")
    print()
