import numpy as np
import pandas as pd
import torchtext
import zipfile
import pathlib 
from pathlib import Path
import os


data_path=Path('data')
data_path.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile('data/archive.zip','r') as zip_ref:
    zip_ref.extractall(data_path)

# Loading the "train data"
df = pd.read_csv('data/twitter_training.csv',header=None)
df.fillna("",inplace=True)

# Rename the column with index 3 to "tweet"
df = df.rename(columns={3: "tweet"})

# Convert the column with index 2 to a categorical data type and assign it to a new column called "sentiment"
df['sentiment'] = df[2].astype('category')

# Add a numerical label to the sentiment column using pandas' categorical codes
df["label"] = df["sentiment"].cat.codes

# Save selected columns of the dataframe to a csv file
train_cleaned_path = data_path / "train_cleaned.csv"

df[['tweet', 'sentiment', 'label']].to_csv(train_cleaned_path, index=False)

