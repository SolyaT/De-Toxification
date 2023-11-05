import os
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import re


def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text


def preprocess_data(df):
    df = df.dropna()
    df['reference'] = df['reference'].apply(clean_text)
    df['translation'] = df['translation'].apply(clean_text)
    return df


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


def save_data(df, file_name, save_dir='data/interim'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(os.path.join(save_dir, file_name), index=False)


def main():
    raw_data_path = '../../data/raw/filtered.tsv'
    train_file = 'train.csv'
    val_file = 'val.csv'
    test_file = 'test.csv'

    df = load_data(raw_data_path)
    df = preprocess_data(df)
    train_df, val_df, test_df = split_data(df)
    save_data(train_df, train_file)
    save_data(val_df, val_file)
    save_data(test_df, test_file)
    print("Data preprocessing and split complete.")


if __name__ == '__main__':
    main()
