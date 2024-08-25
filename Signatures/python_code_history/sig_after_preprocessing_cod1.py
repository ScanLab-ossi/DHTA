import pandas as pd
import glob
import calcs   as calcs
import os
import Signatures.python_code.distances_sockpuppet as distances_sockpuppet

current_path = os.getcwd()

# Step 1 - merge local+global (outer) into 'concat_output.csv' 
# 
# global file content  = Top 100 most prevalent words the words (global) , 

globaldf=pd.read_csv(current_path + '/Signatures/results_cod/all_global_freq.csv')
localdf =pd.read_csv(current_path + '/Signatures/results_cod/all_local_freq.csv')
concat_df = pd.merge(localdf,globaldf, on='word', how='left' )  

 
# Step 5: Save the final merged DataFrame to a new CSV file
concat_df.to_csv(current_path + '/Signatures/results_cod/concat_output.csv', index=False)

print("Merging complete. Output saved to 'concat_output.csv'.")

 
import numpy as np
 
# Define the columns for signatures raw 
sig_columns = ['doc','word','wordavg','percent_of_total','JSD','kl_distance','dt']  

# Create an empty DataFrame with the specified columns
sig_df = pd.DataFrame(columns=sig_columns)
#final_df = pd.DataFrame(columns=final_columns)

# Working on Each document 

unique_files = concat_df['doc'].unique()
 
results_sig_df= pd.DataFrame(columns=['doc_id','word', 'JSD']) 

for doc  in unique_files:  #loop per year 
    filtered_df = concat_df[concat_df['doc'] == doc]  # df only for 1 year = vector
    for index, row in filtered_df.iterrows():
        p=row['wordavg']  #Constant All the COD's and years
        q = row['percent_of_total']  # list of All the COD's in specific year  
        klpart2=  row['wordavg']/row['percent_of_total']  
        kl_distance =  (row['wordavg'] -row['percent_of_total'] ) *   np.log10(klpart2)
        kl_distance =    np.log10(klpart2)

        js_distance = calcs.calculate_jsd_sig_distance(p, q)  # return 1 value per year per cod 
        #js_distance_general=calcs.calculate_js_distance(p,q)
        # append the results to the dataframe by location  
        new_row = pd.DataFrame({'doc_id': [doc], 'JSD': [js_distance],'word':[row['word']], 'kl_distance':[ kl_distance ] }) 
        print(new_row) 
        results_sig_df =  pd.concat([results_sig_df,new_row], ignore_index=True)


    print (results_sig_df)    
 
    

 

# write the results of signatures into CSV

results_sig_df.to_csv(current_path+'/Signatures/results_cod/signatures_30.csv', index=True)  
 
# Create distances matrix for frequencies 

 

# distances Sockpuppet
from scipy.spatial.distance import pdist, squareform 
import numpy as np

# input is original frequencies vector 

summary = localdf.groupby(['doc_id','word'])['percent_of_total'].agg(['sum']).reset_index() 
distances_sockpuppet.sockpuppet(summary,'Orig','euclidean') 
distances_sockpuppet.sockpuppet(summary,'local-JSD','jensenshannon') 

# input is JSD results vector 

summary = results_sig_df.groupby(['doc_id','word'])['JSD'].agg(['sum']).reset_index() 
distances_sockpuppet.sockpuppet(summary,'jensenshannon','jensenshannon') 
distances_sockpuppet.sockpuppet(summary,'jensenshannon-euc','euclidean')  
"""
#
 
summary = sig_df.groupby(['doc_id','word'])['JSD'].agg(['sum']).reset_index() 
distances_sockpuppet.sockpuppet(summary,'JSD') 
 

summary = localdf.groupby(['doc_id','word'])['percent_of_total'].agg(['sum']).reset_index() 
data=distances_sockpuppet.sp_altair(summary,'Altair') 
print(data)
print("Row Index Names:", data.index)
print("Column Names:", data.columns)
"""
"""
import altair as alt
heatmap = alt.Chart(data).mark_rect().encode(
    x=str('X:O'),
    y=str('Y:O'),
    color='Value:Q',
    tooltip=['X', 'Y', 'Value']  # Tooltips to show values on hover
).properties(
    width=300,
    height=300,
    title="Heatmap Example"
)

heatmap.show()
"""
"""
# Calculate pairwise cityblock distance
 
summary = localdf.groupby('doc_id')['percent_of_total'].agg(['sum']) 
#summary = summary.set_index('year')
summary=(summary.head(30))  
print(summary)  


distances = pdist(summary, metric='cityblock')
distance_matrix = squareform(distances)
distance_df = pd.DataFrame(distance_matrix, index=summary.index, columns=summary.index)

print(distance_df)
distance_df.to_csv(current_path + '/Signatures/results_cod/cityblock_distance_df.csv', index=False)

# Get the indices for the lower triangle, excluding the diagonal (k=0)
lower_triangle_indices = np.tril_indices_from(distance_df, k=-1)

# Set the lower triangle values to NaN while keeping the diagonal intact
distance_df.values[lower_triangle_indices] = np.nan

print("\nModified Distance Matrix with Lower Triangle Removed (Diagonal Intact):")
print(distance_df)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
plt.figure(figsize=(10, 8))
sns.heatmap(distance_df, annot=False, cmap='Greens',  square=True,linewidths=0.5, linecolor='gray',annot_kws={"size": 8} ) 
plt.title("Original data")
plt.show()

# Calculate distances between JSD's 

 
# Calculate pairwise cityblock distance
 
summary = sig_df.groupby('doc_id')['JSD'].agg(['max']) 
#summary = summary.set_index('year')
summary=(summary.head(30))  
print(summary)  


distances = pdist(summary, metric='cityblock')
distances = pdist(summary, metric='euclidean')

distance_matrix = squareform(distances)
distance_df = pd.DataFrame(distance_matrix, index=summary.index, columns=summary.index)

print(distance_df)
distance_df.to_csv(current_path + '/Signatures/results_cod/cityblock_sig_df_distance_df.csv', index=False)

# Get the indices for the lower triangle, excluding the diagonal (k=0)
lower_triangle_indices = np.tril_indices_from(distance_df, k=-1)

# Set the lower triangle values to NaN while keeping the diagonal intact
distance_df.values[lower_triangle_indices] = np.nan

print("\nModified Distance Matrix with Lower Triangle Removed (Diagonal Intact):")
print(distance_df)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
plt.figure(figsize=(10, 8))
sns.heatmap(distance_df, annot=False, cmap='Greens',  square=True,linewidths=0.5, linecolor='gray',annot_kws={"size": 8} ) 
plt.title("JSD data")
plt.show()

import altair as alt

distance_df = pd.DataFrame(distance_matrix, index=summary.index, columns=summary.index)

print(distance_df) 
heatmap = alt.Chart(distance_matrix) 
""" 
