# Classification with KNN
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def knn_classification(df, distance_metric='euclidean'):
    print(f"\n--- Classification with {distance_metric.capitalize()} Metric ---")

    X = np.stack(df['embedding'].values)
    y = df['syndrome_id']
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_k, best_score = 0, 0
    
    # Testing K from 1 to 15
    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        scores = cross_val_score(knn, X, y, cv=skf, scoring='f1_weighted')
        mean_score = np.mean(scores)
        print(f'K={k}, Mean F1-Score: {mean_score}')
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f'\nBest K: {best_k} with F1-Score: {best_score}')

    return best_k, best_score
