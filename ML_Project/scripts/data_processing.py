#---DATA PREPROCESSING FOR SYNDROME CLASSIFICATION---

import pickle
import numpy as np
import pandas as pd

def load_and_process_data():
    print("\n--- Loading and Processing Data ---")
    file_path = '../data/mini_gm_public_v0.1.p'
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Flattening the Hierarchical Structure
    flat_data = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                flat_data.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })

    df = pd.DataFrame(flat_data)
    print("DataFrame Structure:")
    print(df.head())

    # 'embedding' Column to NumPy Array
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
    
    df.to_pickle('../data/flattened_data.pkl')
    print("Data loaded and flattened successfully!")
    return df
