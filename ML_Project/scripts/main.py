# ACTIVATION OF THE COMPLETE PIPELINE FOR GENETIC SYNDROME CLASSIFICATION
import sys
import os
from report_generator import generate_report
from data_processing import load_and_process_data
from exploratory_data_analysis import exploratory_data_analysis
from visualization import tsne_visualization
from classification import knn_classification
from metrics_evaluation import evaluate_metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def compare_models(results):
    print("\n--- Model Comparison ---")
    print("\n| Metric            | Euclidean | Cosine |")
    print("|------------------|------------|--------|")
    print(f"| Best K            | {results['euclidean']['best_k']}          | {results['cosine']['best_k']}       |")
    print(f"| F1-Score          | {results['euclidean']['f1_score']:.2f}     | {results['cosine']['f1_score']:.2f} |")
    print(f"| AUC ROC           | {results['euclidean']['roc_auc']:.2f}     | {results['cosine']['roc_auc']:.2f} |")
    print(f"| Top-k Accuracy    | {results['euclidean']['top_k_acc']:.2f}   | {results['cosine']['top_k_acc']:.2f} |")

if __name__ == '__main__':
    print("\n--- Pipeline Start ---")

    # Data Preprocessing
    df = load_and_process_data()

    # Exploratory Data Analysis
    exploratory_data_analysis(df)

    # t-SNE Visualization
    tsne_visualization(df)

    # Classification and Evaluation - Euclidean
    print('\n--- Classification with Euclidean Metric ---')
    best_k_euclidean, f1_score_euclidean = knn_classification(df, distance_metric='euclidean')
    roc_auc_euclidean, top_k_acc_euclidean = evaluate_metrics(df, distance_metric='euclidean')

    # Classification and Evaluation - Cosine
    print('\n--- Classification with Cosine Metric ---')
    best_k_cosine, f1_score_cosine = knn_classification(df, distance_metric='cosine')
    roc_auc_cosine, top_k_acc_cosine = evaluate_metrics(df, distance_metric='cosine')

    # Model Comparison
    results = {
        'euclidean': {
            'best_k': best_k_euclidean,
            'f1_score': f1_score_euclidean,
            'roc_auc': roc_auc_euclidean,
            'top_k_acc': top_k_acc_euclidean
        },
        'cosine': {
            'best_k': best_k_cosine,
            'f1_score': f1_score_cosine,
            'roc_auc': roc_auc_cosine,
            'top_k_acc': top_k_acc_cosine
        }
    }
    compare_models(results)

    generate_report(results)

    print("\n--- Pipeline End ---")
