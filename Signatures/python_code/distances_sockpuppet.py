import pandas as pd
import glob
import calcs   as calcs
import os
from scipy.spatial.distance import pdist, squareform 
import numpy as np

current_path = os.getcwd() 
 

# distances Sockpuppet


def sockpuppet(summary,graph_title,pdist_type):
    import pandas as pd
    print (summary.head())
    X = summary['word'].unique()
    Y = summary['doc_id'].unique()
    #pdist_type='jensenshannon'

    print(X,Y) 
    # create a new dataframe for all the combinations of X,Y
    # Pivot the data to get a matrix
    pivot_df = summary.pivot(index='doc_id', columns='word', values='sum').fillna(0)
    print(pivot_df)


    """
    Explain : 
        pdist(pivot_df.values, metric='cityblock'): Computes the pairwise distances using the chosen metric (e.g., Manhattan distance).
        squareform(distance_matrix): Converts the condensed distance matrix into a square matrix.
    """
    distance_matrix = pdist(pivot_df.values, metric=pdist_type)  #'euclidean')   
    # Convert to a squareform distance matrix for easier readability
    distance_df = pd.DataFrame(squareform(distance_matrix), index=Y, columns=Y)

    # Save results 
    distance_df.to_csv(current_path+'/Signatures/results_cod/'+pdist_type+'.csv', index=True)  

    print(distance_df)

    # Get the indices for the lower triangle, excluding the diagonal (k=0)
    lower_triangle_indices = np.tril_indices_from(distance_df, k=-1)

    # Set the lower triangle values to NaN while keeping the diagonal intact
    distance_df.values[lower_triangle_indices] = np.nan

    # Plot 

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd 
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_df, annot=False, cmap='Greens',  square=True,linewidths=0.5, linecolor='gray',annot_kws={"size": 8} ) 
    plt.title(graph_title)
    # Save the figure to a file
    plt.savefig(current_path+'/Signatures/results_cod/'+graph_title+'.png', dpi=300, bbox_inches='tight')
    plt.show() 

def sp_altair(summary,graph_title):
    import pandas as pd
    print (summary.head())
    X = summary['word'].unique()
    Y = summary['doc_id'].unique()

    print(X,Y) 
    # create a new dataframe for all the combinations of X,Y
    # Pivot the data to get a matrix
    pivot_df = summary.pivot(index='doc_id', columns='word', values='sum').fillna(0)
    print(pivot_df)


    """
    Explain : 
        pdist(pivot_df.values, metric='cityblock'): Computes the pairwise distances using the chosen metric (e.g., Manhattan distance).
        squareform(distance_matrix): Converts the condensed distance matrix into a square matrix.
    """
    distance_matrix = pdist(pivot_df.values, metric='cityblock')   
    # Convert to a squareform distance matrix for easier readability
    distance_df = pd.DataFrame(squareform(distance_matrix))
    return(distance_df)