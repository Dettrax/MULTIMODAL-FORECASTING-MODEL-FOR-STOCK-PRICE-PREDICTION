from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from newspaper import Article
import pandas as pd
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')

service = Service(executable_path='./chromedriver.exe')

driver = webdriver.Chrome(service=service)

df = pd.DataFrame(columns=['Title', 'URL', 'Publish Date', 'Content'])

for pages in tqdm(range(200,326)):
    time.sleep(1)
    wait = WebDriverWait(driver, 5)
    driver.get(f"https://oilprice.com/search/tab/articles/brent_oil/Page-{pages}.html")
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#search-results-articles a')))
    #driver.execute_script("window.stop(); alert('Content is located, loading stopped!')")
    elements = driver.find_elements(By.CSS_SELECTOR,'#search-results-articles a')
    dates = driver.find_elements(By.CSS_SELECTOR,'.dateadded')

    for elem, date in zip(elements, dates):
        url = elem.get_attribute('href')
        d = date.text
        # Create an Article object
        article = Article(url)
        time.sleep(0.2)
        # Download and parse the article
        article.download()
        article.parse()
        data = {
            'Title': article.title,
            'URL': url,
            'Publish Date': d,
            'Content': article.text
        }
        # Convert the dictionary into a DataFrame
        article_df = pd.DataFrame(data, index=[0])

        # Append the article data to the main DataFrame
        df = pd.concat([df, article_df], ignore_index=True)

driver.quit()

df['Publish Date'] = pd.to_datetime(df['Publish Date'], format='%d %B %Y')

df.to_excel('brent300.xlsx', index=False)

#df = pd.read_excel('crude74.xlsx')
#
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')
#
# # Create a SentimentIntensityAnalyzer object
# sia = SentimentIntensityAnalyzer()
#
# # Apply the polarity_scores method to the 'Content' column
# sentiment_scores = df['Content'].apply(sia.polarity_scores)
#
# # Convert the list of dictionaries into a DataFrame
# sentiment_df = pd.DataFrame(list(sentiment_scores))
#
# # Concatenate the original DataFrame with the sentiment scores DataFrame
# df = pd.concat([df, sentiment_df], axis=1)
#
# import matplotlib.pyplot as plt
#
# # Create a histogram of the 'compound' scores
# plt.hist(df['compound'], bins=20, edgecolor='black')
#
# # Set the title and labels
# plt.title('Distribution of Compound Scores')
# plt.xlabel('Compound Score')
# plt.ylabel('Frequency')
#
# # Show the plot
# plt.show()
#
# def categorize_sentiment(score):
#     if score <= -0.25:
#         return 'Negative'
#     elif score >= 0.25:
#         return 'Positive'
#     else:
#         return 'Neutral'
#
# # Apply the categorize_sentiment function to the 'compound' column
# df['compound_category'] = df['compound'].apply(categorize_sentiment)
#
# # Count the number of articles in each category
# compound_count = df['compound_category'].value_counts()
#
# print('Compound count:\n', compound_count)
