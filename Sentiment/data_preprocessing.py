"""
Based on: https://www.kaggle.com/code/turankeskin/twittersentanaly
"""

import numpy as np
import pandas as pd
import re
import string

import os
for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

columns = ["Id","Entity","Target","Text"]

for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        data = pd.read_csv(os.path.join(dirname, filename),
                           names=columns,header=None, skipinitialspace = True)

        df = data[["Text","Target","Id"]]
        df = df.drop_duplicates(subset=['Text', 'Target'])
        df = df.dropna(subset=['Text'])

        sentiment = []

        for i in df["Target"]:
            if i == "Positive":
                sentiment.append(3)
            elif (i == "Irrelevant") or (i == "Neutral"):
                sentiment.append(2)
            else:
                sentiment.append(1)

        df["Sentiment"] = sentiment

        #df["Text"] = df['Text'].astype(str).str.replace("\d", "")
        #df.replace('\n', ' ', regex=True)
        df["Text"] = df['Text'].astype(str).str.replace("\n", "")

        #df["Text"] = df["Text"].str.replace("\d", "")

        del df["Target"]
        df = df.reindex(columns=['Text', 'Sentiment', 'Id'])

        # Save processed data to csv file
        pre, ext = os.path.splitext(filename)
        fname = pre + '.tsv'
        df.to_csv(os.path.join('data/group', fname), sep='\t', index=False, header=False) # Use Tab to seperate data