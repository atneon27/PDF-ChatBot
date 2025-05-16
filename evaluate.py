import os
import json
import time
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from datasets import Dataset
from sklearn.metrics import ndcg_score
from ragas import evaluate as RagasEvaluate
from ragas.metrics import faithfulness, context_recall, context_precision, answer_relevancy
from app import models, get_conversational_chain

def load_test_dataset():
    with open("data/questions.json", "r") as f:
        data = json.load(f)
    return data

def generate_predictions(qa_data, model_name):
    embeddings = models[model_name]["embedding"]()
    index_path = f"faiss_index/{model_name}"
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    chain = get_conversational_chain(retriever, model_name)

    eval_data = {
        "question": [],
        "ground_truth": [],
        "retrieved_contexts": [],
        "response": []
    }

    for sample in qa_data:
        q = sample["question"]
        gt = sample["answer"]
        retrived_docs = retriever.get_relevant_documents(q)
        retrived_text = [doc.page_content for doc in retrived_docs] 

        response = chain({"question": q}, return_only_outputs=True)
        answer = response["answer"]

        eval_data["question"].append(q)
        eval_data["ground_truth"].append(gt)
        eval_data["retrieved_contexts"].append(retrived_text)
        eval_data["response"].append(answer)
        time.sleep(15)

    return Dataset.from_dict(eval_data)

def evaluate_with_manual_metrics(dataset, k=3):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    reciprocal_ranks = []
    ndcg_scores = []
    
    for gt, retrieved in zip(dataset["ground_truth"], dataset["retrieved_contexts"]):
        retrieved_k = retrieved[:k]
        relevant = [1 if gt in doc else 0 for doc in retrieved_k]

        if not retrieved_k:
            precision_scores.append(0)
            recall_scores.append(0)
            f1_scores.append(0)
            reciprocal_ranks.append(0)
            ndcg_scores.append(0)
            continue
            
        # Precision 
        precision = sum(relevant) / len(retrieved_k)
        precision_scores.append(precision)
        
        # Recall
        recall = 1 if sum(relevant) > 0 else 0
        recall_scores.append(recall)
        
        # F1 score
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        
        # Mean Reciprocal Rank (MRR)
        try:
            first_relevant_idx = relevant.index(1)
            rr = 1 / (first_relevant_idx + 1)
        except ValueError: 
            rr = 0
        reciprocal_ranks.append(rr)
        
        # Normalized Discounted Cummulative Gain
        ideal_ranking = [1] + [0] * (k - 1)
        pred_ranking = relevant + [0] * (k - len(relevant))
        try:
            ndcg = ndcg_score([ideal_ranking], [pred_ranking])
            ndcg_scores.append(ndcg)
        except:
            ndcg_scores.append(0)
    
    return {
        f"precision@{k}": np.mean(precision_scores),
        f"recall@{k}": np.mean(recall_scores),
        f"f1@{k}": np.mean(f1_scores),
        "mrr": np.mean(reciprocal_ranks),
        f"ndcg@{k}": np.mean(ndcg_scores),
    }

def evaluate_with_ragas(dataset):
    ragas_df = RagasEvaluate(dataset, metrics=[
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy
    ]).to_pandas()

    return ragas_df.to_dict("records")

def run_full_evaluation(model_name):
    qa_data = load_test_dataset()
    dataset = generate_predictions(qa_data, model_name)

    print(f"Running Manual Evaluation on {len(dataset)} samples...")
    manual_result = evaluate_with_manual_metrics(dataset)

    print(f"\nRunning RAGAS Evaluation on {len(dataset)} samples...")
    ragas_results = evaluate_with_ragas(dataset)

    manual_summary = { "model": model_name, **manual_result }

    detailed_rows = []
    for i, row in enumerate(dataset):
        entry = {
            "model": model_name,
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "response": row.get("response")
        }

        metrics = ragas_results[i] if i < len(ragas_results) else {}
        entry.update(metrics)

        detailed_rows.append(entry)

    return manual_summary, detailed_rows


if __name__ == "__main__":
    complete_manual_summary = []
    complete_detailed_summary = []

    for model_name in models.keys():
        manual_summary, detailed_rows = run_full_evaluation(model_name)
        complete_manual_summary.append(manual_summary)
        complete_detailed_summary.extend(detailed_rows)

    manual_df = pd.DataFrame(complete_manual_summary)
    manual_df.to_excel("retriever_evals.xlsx", index=False)

    detailed_df = pd.DataFrame(complete_detailed_summary)
    detailed_df.to_excel("generative_evals.xlsx", index=False)
    