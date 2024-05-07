import pandas as pd

import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

crude = pd.read_excel('gn200.xlsx')

for i in [400,600,800,950]:
    crude = pd.concat([crude, pd.read_excel(f'gn{i}.xlsx')], ignore_index=True)

crude = crude[crude['Publish Date'].isnull() == False]


print(crude['Publish Date'].isnull().sum())
print(crude['Title'].isnull().sum())
print(crude['URL'].isnull().sum())
print(crude['Content'].isnull().sum())

crude.reset_index(drop=True, inplace=True)

import dateutil.parser as dparser

print(dparser.parse(str(crude['Publish Date'].loc[0]),fuzzy=True))

crude['Publish Date extracted'] = ''

import re
dates_error = []
for i in range(len(crude)):
    try:
        crude['Publish Date extracted'].loc[i] = dparser.parse(str(crude['Publish Date'].loc[i]),fuzzy=True).date()
    except:
        try:
            match = re.search(r'\b[A-Za-z]{3} \d{1,2}, \d{4} at \d{1,2}:\d{2}\b',str(crude['Publish Date'].loc[i]))
            if match:
                # Extract the date and time
                date_time_str = match.group()
                # Parse the date and time
                date_time = dparser.parse(date_time_str,fuzzy=True).date()
                crude['Publish Date extracted'].loc[i] = date_time
        except:
            dates_error.append(i)

crude['Publish Date'] = crude['Publish Date extracted']
crude.drop(columns=['Publish Date extracted'], inplace=True)

brent = pd.read_csv('brent_processed.csv')

brent.index = pd.to_datetime(brent['Date'], format='%Y-%m-%d')

brent_dates = pd.DataFrame(brent.index)

brent_dates['Date'] = brent_dates['Date'].apply(lambda x: dparser.parse(str(x),fuzzy=True).date())

# Convert both columns to sets
brent_dates_set = set(brent_dates['Date'])
crude_dates_set = set(crude['Publish Date'])

# Find dates that are in brent_dates but not in crude
dates_in_brent_not_in_crude = brent_dates_set - crude_dates_set

# Convert the result back to a list
dates_in_brent_not_in_crude = list(dates_in_brent_not_in_crude)

# Convert both columns to sets
brent_dates_set = set(brent_dates['Date'])
crude_dates_set = set(crude['Publish Date'])

# Find dates that are in both brent_dates and crude
dates_in_both = brent_dates_set & crude_dates_set

# Convert the result back to a list
dates_in_both = list(dates_in_both)

#2165 and 750
#918 and 1997

dates_in_brent_not_in_crude = pd.DataFrame(dates_in_brent_not_in_crude, columns=['Date'])

#sort bt dates
dates_in_brent_not_in_crude.sort_values(by='Date', inplace=True)

#2013-01-01 - 2013-12-31

# Remove duplicates based on 'Date'
crude = crude.drop_duplicates(subset='Title', keep='first')
# crude = crude.drop_duplicates(subset='Publish Date', keep='first')

# Reset the index after dropping rows
crude.reset_index(drop=True, inplace=True)

# crude.to_excel('crude_test.xlsx', index=False)


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure the necessary NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the stop words
stop_words = set(stopwords.words('english'))


# Define a function to handle the preprocessing
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove non-alphabetic tokens and convert to lower case
    words = [word.lower() for word in tokens if word.isalpha()]

    # Remove the stop words
    words = [word for word in words if not word in stop_words]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    return words


# Apply the preprocessing to the 'Content' column
crude['Content'] = crude['Content'].apply(preprocess_text)

crude['Content'] = [' '.join(map(str, l)) for l in crude['Content']]

crude.to_excel('crude_test.xlsx', index=False)