import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from  tqdm import tqdm
# Load the data
crude = pd.read_excel('crude_test.xlsx')
crude.reset_index(drop=True, inplace=True)
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone',add_special_tokens=False)
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
finbert.to(device)

# Initialize the sentiment analysis pipeline
nlp = pipeline('sentiment-analysis', model=finbert, device=0, tokenizer=tokenizer)

# Initialize an empty list to hold the sentiment scores
sentiment_scores = []

# Iterate over each text in the 'Content' column of the 'crude' DataFrame
for i in tqdm(range(len(crude))):
    text = crude['Content'].loc[i]
    # Tokenize the text and truncate it to the maximum length
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:510]  # Truncate the tokens
    text = tokenizer.convert_tokens_to_string(tokens)  # Convert the tokens back into a string

    # Perform sentiment analysis on the text
    result = nlp(text)
    # Append the sentiment score to the list
    if result[0]['label'] == 'Negative':
       sentiment_scores.append({'pos': 0, 'neg': result[0]['score'], 'neu': 0})
    elif 'pos' in result[0]['label'].lower():
        sentiment_scores.append({'pos': result[0]['score'], 'neg': 0, 'neu': 0})
    else:
        sentiment_scores.append({'pos': 0, 'neg': 0, 'neu': result[0]['score']})

# Convert the list of sentiment scores to a DataFrame
sentiment_scores_df = pd.DataFrame(sentiment_scores)

# Concatenate the 'crude' DataFrame with the 'sentiment_scores_df' DataFrame
crude = pd.concat([crude, sentiment_scores_df], axis=1)

crude.to_excel('final_df.xlsx', index=False)

