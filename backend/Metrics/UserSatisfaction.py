import pandas as pd

def feedback_score(feedback_column):
    total = len(feedback_column)
    positive = feedback_column.eq(1).sum()
    if total == 0:
        return 0.0
    return (positive / total) * 5  # 5-point scale

def write_feedback_benchmark(scores, path="../logs/UserFeedback_benchmark.txt"):
    with open(path, "w") as f:
        for label, score in scores.items():
            f.write(f"{label} User Feedback Score: {score:.2f} / 5.00\n")

if __name__ == "__main__":
    files = {
        "AI ChatBot": "../logs/queries_responses_ai.xlsx",
        "Concordia ChatBot": "../logs/queries_responses_concordia.xlsx",
        "General Assistant ChatBot": "../logs/queries_responses_general.xlsx"
    }

    results = {}
    for label, path in files.items():
        df = pd.read_excel(path)
        feedback_col = df.iloc[:, 2]
        score = feedback_score(feedback_col)
        results[label] = score
        print(f"{label} User Feedback Score: {score:.2f} / 5.00")

    write_feedback_benchmark(results)
