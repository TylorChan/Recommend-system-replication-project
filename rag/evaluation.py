import pickle


for each in ["reddit", "inspired", "redial"]:
    print(each)
    test_data ={}

    with open(f"pred_result_{each}.pickle", "rb") as file:
        test_data = pickle.load(file)

    count1 = 0
    count5 = 0
    count20 = 0

    for idx, each in enumerate(test_data["label"]):
        if each == test_data["pred_label"][idx][0]:
            count1 += 1
        if each in test_data["pred_label"][idx][:5]:
            count5 += 1
        if each in test_data["pred_label"][idx][:20]:
            count20 += 1  
    
    hit1 = (count1/ len(test_data["pred_label"])) * 100
    hit5 = (count5/ len(test_data["pred_label"])) * 100
    hit20 = (count20/ len(test_data["pred_label"])) * 100

    print("hit@1:", f"%{hit1}")
    print("hit@5:", f"%{hit5}")
    print("hit@20:", f"%{hit20}")
    print()



for each in ["reddit", "inspired", "redial"]:
    print(each, "rag")
    test_data ={}

    with open(f"pred_result_{each}_rag.pickle", "rb") as file:
        test_data = pickle.load(file)

    count1 = 0
    count5 = 0
    count20 = 0

    for idx, each in enumerate(test_data["label"]):
        if each == test_data["pred_label"][idx][0]:
            count1 += 1
        if each in test_data["pred_label"][idx][:5]:
            count5 += 1
        if each in test_data["pred_label"][idx][:20]:
            count20 += 1  
    
    hit1 = (count1/ len(test_data["pred_label"])) * 100
    hit5 = (count5/ len(test_data["pred_label"])) * 100
    hit20 = (count20/ len(test_data["pred_label"])) * 100

    print("hit@1:", f"%{hit1}")
    print("hit@5:", f"%{hit5}")
    print("hit@20:", f"%{hit20}")
    print()