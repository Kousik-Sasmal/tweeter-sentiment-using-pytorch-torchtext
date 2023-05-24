import numpy as np
import pandas as pd
import zipfile
import os

# initializing `artifacts` folder
os.makedirs('artifacts',exist_ok=True)

try:
    with zipfile.ZipFile('notebooks/data/archive.zip','r') as zip_ref:
        zip_ref.extractall('notebooks/data')
except Exception as e:
    print(e)

# Loading the "train data"
df = pd.read_csv('notebooks/data/twitter_training.csv',header=None)
df.dropna(inplace=True)



# Rename the column with index 3 to "tweet"
df = df.rename(columns={3: "tweet"})
df['tweet'] = df['tweet'].str.lower()
df.drop_duplicates(inplace=True)


# Convert the column with index 2 to a categorical data type and assign it to a new column called "sentiment"
df['sentiment'] = df[2].astype('category')

# Add a numerical label to the sentiment column using pandas' categorical codes
df["label"] = df["sentiment"].cat.codes

# Save selected columns of the dataframe to a csv file
train_cleaned_path = "artifacts/train_cleaned.csv"

df[['tweet', 'sentiment', 'label']].to_csv(train_cleaned_path, index=False)


#---------------------------------------------------------------------------#


# Loading the "validation data"
df_valid = pd.read_csv('notebooks/data/twitter_validation.csv',header=None)
df_valid.dropna(inplace=True)


# Rename the column with index 3 to "tweet"
df_valid = df_valid.rename(columns={3: "tweet"})
df_valid['tweet'] = df_valid['tweet'].str.lower()
df_valid.drop_duplicates(inplace=True)


# Convert the column with index 2 to a categorical data type and assign it to a new column called "sentiment"
df_valid['sentiment'] = df_valid[2].astype('category')

# Add a numerical label to the sentiment column using pandas' categorical codes
df_valid["label"] = df_valid["sentiment"].cat.codes

# Save selected columns of the dataframe to a csv file
valid_cleaned_path = "artifacts/valid_cleaned.csv"

df_valid[['tweet', 'sentiment', 'label']].to_csv(valid_cleaned_path, index=False)