# data_preparation.py
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_dpr_data():
    # Load SubjectQA dataset (electronics domain) with custom code execution enabled
    dataset = datasets.load_dataset("subjqa", name="electronics", trust_remote_code=True)
    
    # Flatten all splits and combine into one DataFrame
    dfs = []
    for split, ds in dataset.flatten().items():
        df = ds.to_pandas()
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates using a unique identifier such as 'id'
    full_df = full_df.drop_duplicates(subset=["id"])
    
    # Save the full dataset for reference
    full_df.to_csv("subjectqa_full.csv", index=False)
    
    # Create train, test, and validation splits:
    # 70% training, 15% test, 15% validation.
    train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df.to_csv("subjectqa_train.csv", index=False)
    test_df.to_csv("subjectqa_test.csv", index=False)
    val_df.to_csv("subjectqa_val.csv", index=False)
    
if __name__ == "__main__":
    prepare_dpr_data()
