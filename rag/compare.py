import pickle


for each in ["reddit", "inspired", "redial"]:
    print(each)

    with open(f"pred_result_{each}.pickle", "rb") as file:
        retriever_data = pickle.load(file)
    
    with open(f"pred_result_{each}_rag.pickle", "rb") as file:
        ranker_data = pickle.load(file)

    for each_retriever_data, each_ranker_data in zip(retriever_data["pred_label"], ranker_data["pred_label"]):
        if set(each_retriever_data) != set(each_ranker_data):
            print("found")
            print(each_retriever_data)
            print(each_ranker_data)
            break
