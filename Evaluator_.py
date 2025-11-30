from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def evaluate_summaries(articles, summaries):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rougeL": []}

    for article, summary in zip(articles, summaries):
        score = scorer.score(article, summary)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)

    return scores

def plot_rouge_scores(scores):
    plt.figure(figsize=(7, 5))
    plt.plot(scores["rouge1"], marker="o", label="ROUGE-1 F1")
    plt.plot(scores["rougeL"], marker="s", label="ROUGE-L F1")
    plt.xlabel("News Article Index")
    plt.ylabel("F1 Score")
    plt.title("ROUGE Scores for Summaries")
    plt.legend()
    plt.grid(True)
    plt.savefig("rouge_scores.png")
    plt.show()

def save_result(article, summary_en, summary_ko, path="results.csv"):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), article, summary_en, summary_ko])
