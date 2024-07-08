
import datetime
import pandas
from datetime import datetime
import time
import csv
import pandas as pd
import os



# Load the files
directory = '/Users/kash/Desktop/nlp/finalproject/Tesla_SP'
listdir=os.listdir(directory)


# Load the first dataframe
df1 = pd.read_csv('/Users/kash/Desktop/nlp/finalproject/articles.csv')


#concatenating all the historical data into one file. 
def merge_historical():
    df_list=[]
    for file_name in listdir:
        name= directory +'/'+ file_name
        df = pd.read_csv(name)
        df['Date'] = pd.to_datetime(df1['Date'])
        df_list.append(df)
    


    # Concatenate the dataframes based on date
    merged_data = pd.concat(df_list)
    merged_data = merged_data.sort_values('Date')

    # Save the merged data to a CSV file
    merged_data.to_csv('merged_data.csv', index=False)










# Load the second dataframe
df2 = pd.read_csv('/Users/kash/Desktop/nlp/finalproject/merged_data.csv')

# Convert "Date" column to datetime format
df1['Date'] = pd.to_datetime(df2['Date'])
df2['Date'] = pd.to_datetime(df1['Date'])

df1.drop('Time', axis=1)
# Merge the dataframes based on the "Date" column
merged_df = pd.merge(df1, df2, on='Date')

merged_df.to_csv('data1.csv', index=False)








