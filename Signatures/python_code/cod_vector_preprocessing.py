 
import re
import pandas as pd 
import os
 

# Get the current working directory
current_path = os.getcwd()
print(current_path)
"""
with global as 
(select element ,  avg(fr )  global_freq
from 
( SELECT *,sum(value) over (partition by document)  deads, value/sum(value) over (partition by document) fr  FROM `iucc-learning-and-generalizing.cod.normalized_view` )  group by 1 order by 2 desc ) ,
local as 
(SELECT *,sum(value) over (partition by document)  deads, value/sum(value) over (partition by document) local_freq  FROM `iucc-learning-and-generalizing.cod.normalized_view`) 
select document vector , element word,value wordCount, concat ('01/01/',  document) dt ,document  doc_id , document doc,  local_freq  percent_of_total , global_freq  from local left join global using (element)  order by 1,2
"""
# Modify the input file name as required 
filename = current_path+'/Signatures/raw_data/raw_cod.csv'
print (filename) 
df = pd.read_csv(filename)
 
  
# Group by 'word' column and count (DVR)

result = df.groupby('word').agg({  
    'wordCount': 'sum', 
    'percent_of_total':'mean'                # count of 'OtherColumn'
}).rename(columns={ 'wordCount': 'wordCount',  'percent_of_total':'wordavg'})


dfglobal=result
print(dfglobal.head(10))
 
 
# Generate vectors per author 
dflocal = df.groupby(['vector', 'word']).agg({ 
    'wordCount': 'sum', 
    'dt':'max', 
    'doc_id':'max',
    'doc':'max' , 
    'percent_of_total':'mean' 
}).rename(columns={'wordCount': 'wordCount','doc': 'doc','doc_id':'doc_id'  }) 

print(dflocal.head(10))

 


all_local_freq_file_name= current_path+'/Signatures/results_cod/all_local_freq.csv'
dflocal.to_csv(all_local_freq_file_name, index=True)
 

all_global_freq_file_name=current_path+'/Signatures/results_cod/all_global_freq.csv'
dfglobal.to_csv(all_global_freq_file_name, index=True)
 
# Merge dflocal and dfglobal to the same CSV file  
merged_df = pd.merge(dflocal,dfglobal, on='word',   how='left')
#print(merged_df) 
merged_df.to_csv(current_path+'/Signatures/results_cod/agg_merged.csv', index=True)  