import pandas as pd
import glob
import calcs   as calcs
import os

current_path = os.getcwd()

# Step 1 #First file = Top 100 most prevalent words the words (global) 

all_df=pd.read_csv(current_path + '/Signatures/results/all_global_freq.csv')

# Cut top 100 only 

#all_df=all_df.nlargest(1000, 'wordCount') 
all_df.to_csv(current_path + '/Signatures/results/topoutput.csv', index=False)

merged_df = all_df[['word', 'wordavg' , 'wordCount']] 
# Step 2: List all CSV files in the directory
file_list = glob.glob(current_path + '/Signatures/results/data_*.csv')

# Define the columns
columns = ['filename','word', 'wordavg', 'vector', 'Pos', 'wordCount', 'dt', 'doc_id','doc', 'subcorpus', 'percent_of_total']

# Create an empty DataFrame with the specified columns
concat_df = pd.DataFrame(columns=columns)

# Step 4: Iterate and merge each CSV file using an outer join
for i, file in enumerate(file_list):
    # Read the current CSV file
    temp_df = pd.read_csv(file)
    merged_df = all_df[['word', 'wordavg', 'wordCount']] 
    merged_df['filename']=i

    # Merge with the accumulated DataFrame using an outer join
    merged_df = pd.merge(merged_df, temp_df, on='word', how='left')
    concat_df = pd.concat([concat_df, merged_df], axis=0, ignore_index=True) 


# Step 5: Save the final merged DataFrame to a new CSV file
concat_df.to_csv(current_path + '/Signatures/results/concat_output.csv', index=False)

print("Merging complete. Output saved to 'concat_output.csv'.")
import numpy as np

# Define the columns for signatures
sig_columns = ['filename','word','wordavg','percent_of_total','JSD','final_JSD','dt']
#final_columns = ['filename','word','wordavg','percent_of_total','JSD','final_JSD']


# Create an empty DataFrame with the specified columns
sig_df = pd.DataFrame(columns=sig_columns)
#final_df = pd.DataFrame(columns=final_columns)

# Working on Each document 

unique_files = concat_df['filename'].unique()

for filename  in unique_files:

    filtered_df = concat_df[concat_df['filename'] == filename]
    # Extract the values of val1 into a list
    p = filtered_df['wordavg'].tolist()
    q = filtered_df['percent_of_total'].tolist()


    js_distance = calcs.calculate_jsd_sig(p, q)
    js_distance_general=calcs.calculate_js_distance(p,q)
    # append the results to the dataframe by location   

    print(js_distance_general) 

    condition = filtered_df['filename'] ==  filename
    filtered_df.loc[condition,'JSD']=js_distance
    #filtered_df.loc[condition,'final_JSD']=js_distance_general

    #final_df=pd.concat([final_df,filtered_df],axis=0, ignore_index=True)   
    #final_df.loc[condition,'final_JSD']=js_distance_general
    # sort top 30 words for signatures
    top_30_df = filtered_df.nlargest(30, 'JSD')
    selected_columns_df=top_30_df[['filename','word','wordavg','percent_of_total','JSD','wordCount_x','wordCount_y','final_JSD','dt']]
    # Display the selected rows
    print(top_30_df.head(30))
    sig_df = pd.concat([sig_df, top_30_df], axis=0, ignore_index=True)  

 

# write the results of signatures into CSV

sig_df.to_csv(current_path+'/Signatures/results/signatures_30.csv', index=True) 


