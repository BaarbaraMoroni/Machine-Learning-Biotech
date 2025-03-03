# HERE IS PERFORMED THE EVALUATION OF METRICS FOR RESULT COMPARISON

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, top_k_accuracy_score
import matplotlib.pyplot as plt


def evaluate_metrics(df, distance_metric='euclidean'):  # Metrics Evaluation / Generate ROC Curves
    print(f"\n--- Evaluation with {distance_metric.capitalize()} Metric ---")

    X = np.stack(df['embedding'].values)
    y = df['syndrome_id']
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    top_k_acc_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn = KNeighborsClassifier(n_neighbors=15, metric=distance_metric)  # Classification with KNN
        knn.fit(X_train, y_train)
        y_prob = knn.predict_proba(X_test)
        y_pred = knn.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 0], pos_label=knn.classes_[0])  # ROC Curve and AUC
        roc_auc = auc(fpr, tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc)

        top_k_acc = top_k_accuracy_score(y_test, y_prob, k=3)
        top_k_acc_scores.append(top_k_acc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_top_k_acc = np.mean(top_k_acc_scores)

    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'{distance_metric.capitalize()} (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../visuals/roc_auc_{distance_metric}.png')
    plt.close()

    print(f'\nROC AUC ({distance_metric}): {mean_auc}')
    print(f'Top-k Accuracy ({distance_metric}): {mean_top_k_acc}')

    return mean_auc, mean_top_k_acc
