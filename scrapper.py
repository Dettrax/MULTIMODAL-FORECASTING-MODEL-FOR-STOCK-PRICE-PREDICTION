from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from newspaper import Article
import pandas as pd
import requests
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import matplotlib
from tqdm import tqdm
from bs4 import BeautifulSoup
matplotlib.use('TkAgg')

service = Service(executable_path='./chromedriver.exe')

# driver = webdriver.Chrome(service=service)

#950
df = pd.DataFrame(columns=['Title', 'URL', 'Publish Date', 'Content'])
def get_date(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title_element = soup.select_one('.article_byline')
    return title_element.text

from lxml import html,cssselect
for pages in tqdm(range(600,800)):
    time.sleep(1)
    response = requests.get(f"https://oilprice.com/Latest-Energy-News/World-News/Page-{pages}.html")
    tree = html.fromstring(response.content)
    elements = tree.xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "categoryArticle__title", " " ))]/parent::a')
    dates = tree.cssselect('.categoryArticle__meta')
    for elem, date in zip(elements, dates):
        url = elem.get('href')
        d = date.text
        # Create an Article object
        article = Article(url)
        time.sleep(0.2)
        # Download and parse the article
        article.download()
        article.parse()
        if len(str(d)) <1 :
            d = get_date(url)
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



from bs4 import BeautifulSoup
import requests
for i in tqdm(range(len(df))):
    print(i)
    if type(df['Publish Date'].loc[i]) ==  pd._libs.tslibs.nattype.NaTType or df['Publish Date'].loc[i] == '':
        url = df['URL'].loc[i]
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title_element = soup.select_one('.article_byline')
        df.at[i,'Publish Date'] = title_element.text
        print('done')
        time.sleep(0.5)


df.to_excel('gn800.xlsx', index=False)


