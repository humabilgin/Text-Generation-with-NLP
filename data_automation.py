# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:59:58 2022

@author: humab
"""


import pandas as pd
import json


data_dict = [json.loads(line) for line in open('News_Category_Dataset_v2.json', 'r')]

df = pd.DataFrame(data_dict)

filename = []
politics_counter = 1
wellness_counter = 1
entertainment_counter = 1
travel_counter = 1
sports_counter = 1
business_counter = 1

print(df)
print(df.category[0])


for index, row in df.iterrows():
    
    if (row['category'] == 'POLITICS')and(politics_counter<501):
        filename = "politics/p" + str(politics_counter) +".txt"
        politics_counter = politics_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
    elif (row['category'] == 'WELLNESS')and(wellness_counter<501):
        filename = "wellness/w" + str(wellness_counter) +".txt"
        wellness_counter = wellness_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
    elif (row['category'] == 'ENTERTAINMENT')and(entertainment_counter<501):
        filename = "entertainment/e" + str(entertainment_counter) +".txt"
        entertainment_counter = entertainment_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
    elif (row['category'] == 'TRAVEL')and(travel_counter<501):
        filename = "travel/t" + str(travel_counter) +".txt"
        travel_counter = travel_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
    elif (row['category'] == 'SPORTS')and(sports_counter<501):
        filename = "sports/s" + str(sports_counter) +".txt"
        sports_counter = sports_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
    elif (row['category'] == 'BUSINESS')and(business_counter<501):
        filename = "business/b" + str(business_counter) +".txt"
        business_counter = business_counter +1
        f=open(filename, "w", encoding="utf-8")
        f.write(row['headline'])
        
f.close()


    
