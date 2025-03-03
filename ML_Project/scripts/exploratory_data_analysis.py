# HERE IS PERFORMED THE EXPLORATORY ANALYSIS OF SAMPLES BASED ON THE IMPORTED SYNDROMES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def exploratory_data_analysis(df):
    print("General Statistics:")
    print(df.describe())
    print("\nSyndrome Distribution:")
    print(df['syndrome_id'].value_counts())
    
    plt.figure(figsize=(12, 6)) # Syndrome distribution
    sns.countplot(x='syndrome_id', data=df, palette='viridis')
    plt.title('Syndrome Distribution')
    plt.xlabel('Syndrome')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../visuals/syndrome_distribution.png')
    plt.show()

if __name__ == '__main__':
    df = pd.read_pickle('../data/flattened_data.pkl')
    exploratory_data_analysis(df)
