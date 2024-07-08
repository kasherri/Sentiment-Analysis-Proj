import selenium
import datetime
from datetime import datetime
import newspaper
from newspaper import *
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import *
from selenium.common.exceptions import NoSuchElementException
from urllib.error import HTTPError
from selenium.webdriver.common.keys import Keys
import time
import csv






# set up webdriver
driver = webdriver.Chrome()
driver.get("https://www.marketwatch.com/investing/stock/tsla?mod=search_symbol")



element = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div[6]/div[2]/div[1]/mw-tabs/div[2]/div[1]/mw-scrollable-news-v2')


actions = ActionChains(driver)

# Perform a scroll down action on the element

scroll_origin = ScrollOrigin.from_element(element)

for x in range(2000):
    ActionChains(driver)\
            .scroll_from_origin(scroll_origin, 618, 600)\
            .perform()
    time.sleep(0.3)
   
   






"""
name='//*[@id="maincontent"]/div[6]/div[2]/div[1]/mw-tabs/div[2]/div[1]/mw-scrollable-news-v2/div/div/div' + '['+ str(5)+ ']'

article=element.find_element(By.XPATH, name)

article_subclass=article.find_elements(By.CLASS_NAME, "figure__image")

#articless=element.find_elements(By.CLASS_NAME, "article__content")
if not article_subclass:
    print('none')
else:
    article_subclass=article_subclass[0]
    link=article_subclass.get_attribute("href")
    print(link)


"""



"""
def scraper():
    articles={}
    for i in range(1001, 5500):
        
        name='//*[@id="maincontent"]/div[6]/div[2]/div[1]/mw-tabs/div[2]/div[1]/mw-scrollable-news-v2/div/div/div' + '['+ str(i)+ ']'
        
        article=element.find_element(By.XPATH, name)
        article_subclass=article.find_elements(By.CLASS_NAME, "figure__image")
        if not article_subclass:
            continue
        else:
            article_subclass=article_subclass[0]
            link=article_subclass.get_attribute("href")
            time_stamp=article.find_element(By.CLASS_NAME, "article__timestamp")
            date_string=time_stamp.get_attribute("data-est") 
            date_obj=datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S') #final date we will add

            #proccessing text from article
            article_value=Article(link)
            article_value.download()
            article_value.parse()

            if date_obj.date() in articles:
                articles[date_obj.date()].append([date_obj.time(), article_value.text])
            else:
                articles.setdefault(date_obj.date(), []).append([date_obj.time(), article_value.text])

        
        

    return articles



"""

def scraper():
    articles={}
    i=0
    while True:
        try:
            i+=1
            name='//*[@id="maincontent"]/div[6]/div[2]/div[1]/mw-tabs/div[2]/div[1]/mw-scrollable-news-v2/div/div/div' + '['+ str(i)+ ']'
            article = element.find_element(By.XPATH, name)
            article_subclass=article.find_elements(By.CLASS_NAME, "figure__image")
            if not article_subclass:
                continue
            #if link isnt found
            elif not article_subclass[0].get_attribute("href"):
                continue
            else:
                try:
                    article_subclass=article_subclass[0]
                    link=article_subclass.get_attribute("href")
                    time_stamp=article.find_element(By.CLASS_NAME, "article__timestamp")
                    date_string=time_stamp.get_attribute("data-est") 
                    date_obj=datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S') #final date we will add

                    #proccessing text from article
                    article_value=Article(link)
                    article_value.download()
                    article_value.parse()

                    if date_obj.date() in articles:
                        articles[date_obj.date()].append([date_obj.time(), article_value.text])
                    else:
                        articles.setdefault(date_obj.date(), []).append([date_obj.time(), article_value.text])
                except HTTPError as e:
                    continue
                except newspaper.article.ArticleException as e:
                    continue


        except NoSuchElementException:
                # handle the exception, or simply break the loop
                break    
        

    return articles


data=scraper()
with open('articles.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Time', 'Text'])
    for date, events in data.items():
        for event in events:
            writer.writerow([date, event[0], event[1]])











