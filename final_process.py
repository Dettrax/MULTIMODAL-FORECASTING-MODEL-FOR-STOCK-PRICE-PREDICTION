import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
data = pd.read_csv('brent_with_forecasted_volatility_prime.csv')

data = data.dropna(axis=0, how='any')
data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
# data.index = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data =data.drop(['Date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)

data.sort_index(inplace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the BERT model
model = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(device)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

final_df =pd.read_excel('final_df.xlsx')
targets = final_df.iloc[:, 4:]  # sentiment scores

embeddings = []
for i in tqdm(range(len(final_df))):
    text = final_df['Content'].loc[i]
    tokens = tokenizer.encode(text, max_length=510,return_tensors='pt',truncation=True).to(device)
    embeddings.append(model(tokens)[0][0][0].detach().cpu().numpy())

features = pd.DataFrame(embeddings)

features = pd.concat([features,final_df[['pos','neg','neu']]], axis=1)

# Create a list of column names
column_names = ['vec_' + str(i) for i in range(features.shape[1]-3)]
column_names += ['pos', 'neg', 'neu']
# Assign the column names to the DataFrame
features.columns = column_names

features.index = final_df['Publish Date']

features_grouped = features.groupby(features.index).mean()

#concat features and final_df

missing_dates = data.index[~data.index.isin(features_grouped.index)]

# Merge the data and features_grouped DataFrames
merged_data = data.merge(features_grouped, how='left', left_index=True, right_index=True)

#check empty values in the merged_data
print(merged_data.isnull().sum().any())

# Fill missing values with the previous row's data
merged_data.fillna(method='ffill', inplace=True)

merged_data.to_excel('brent_vec.xlsx', index=False)
