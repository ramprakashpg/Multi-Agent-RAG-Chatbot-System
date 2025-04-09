import pandas as pd

def calculate_accuracy(df):
    correct = df['Feedback'].eq(1).sum()
    total = len(df)
    return correct / total if total else 0

def write_benchmark(entries, path="../logs/Accuracy_benchmark.txt"):
    with open(path, "w") as f:
        for label, accuracy in entries.items():
            f.write(f"{label} Accuracy: {accuracy:.2%}\n")

if __name__ == "__main__":
    # file paths
    files = {
        "AI ChatBot": "../logs/queries_responses_ai.xlsx",
        "Concordia ChatBot": "../logs/queries_responses_concordia.xlsx",
        "General Assistant ChatBot": "../logs/queries_responses_general.xlsx"
    }

    # Collect results
    results = {}
    for label, path in files.items():
        df = pd.read_excel(path)
        accuracy = calculate_accuracy(df)
        results[label] = accuracy
        print(f"{label} Accuracy: {accuracy:.2%}")

    # Write all results to the file
    write_benchmark(results)
