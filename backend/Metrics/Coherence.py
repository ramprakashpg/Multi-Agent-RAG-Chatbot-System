import pandas as pd
from sentence_transformers import SentenceTransformer, util

#Calculate the Cosine Similairty
def calculate_average_similarity(queries, responses, model):
    similarities = []
    for query, response in zip(queries, responses):
        if pd.notna(query) and pd.notna(response):
            emb1 = model.encode(str(query), convert_to_tensor=True)
            emb2 = model.encode(str(response), convert_to_tensor=True)
            sim = util.cos_sim(emb1, emb2).item()
            similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0.0

def write_coherence_benchmark(entries, path="../logs/Coherence_benchmark.txt"):
    with open(path, "w") as f:
        for label, similarity in entries.items():
            f.write(f"{label} Coherence (Cosine Similarity): {similarity:.2%}\n")

if __name__ == "__main__":
    files = {
        "AI ChatBot": "../logs/queries_responses_ai.xlsx",
        "Concordia ChatBot": "../logs/queries_responses_concordia.xlsx",
        "General Assistant ChatBot": "../logs/queries_responses_general.xlsx"
    }

    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = {}

    for label, path in files.items():
        df = pd.read_excel(path)
        queries = df.iloc[:, 0]
        responses = df.iloc[:, 1]
        similarity = calculate_average_similarity(queries, responses, model)
        results[label] = similarity
        print(f"{label} Coherence: {similarity:.2%}")

    write_coherence_benchmark(results)
