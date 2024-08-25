import pandas as pd
import glob
import calcs   as calcs
import os
import distances_sockpuppet as distances_sockpuppet
import numpy as np

#alias='cod'
current_path = os.getcwd()
def sig(alias): 
    import numpy as np

 
    # Step 1 - merge local+global (outer) into 'concat_output.csv' 
    # 
    # global file content  = Top 100 most prevalent words the words (global) , 

    globaldf=pd.read_csv(current_path + '/Signatures/results_'+alias+'/all_global_freq.csv')
    localdf =pd.read_csv(current_path + '/Signatures/results_'+alias+'/all_local_freq.csv')
    concat_df = pd.merge(localdf,globaldf, on='word', how='left' )  

    
    # Step 5: Save the final merged DataFrame to a new CSV file
    concat_df.to_csv(current_path + '/Signatures/results_'+alias+'/concat_output.csv', index=False)

    print("Merging complete. Output saved to 'concat_output.csv'.")

 
 
    # Define the columns for signatures raw 
    sig_columns = ['doc','word','wordavg','percent_of_total','JSD','dt', 'H','I','J','K','L','M']  
    # H local   I  global   
    # M	KL(G,M)	KL(H,M)	jsd
    # =(H2+I2)/2     =LN((H2/J2))*H2     =LN((I2/J2))*I2     =0.5*((K2)+(L2))
    #0.003220002	0.00065547	-0.000542126	0.0000567
    # Create an empty DataFrame with the specified columns
    sig_df = pd.DataFrame(columns=sig_columns)
    #final_df = pd.DataFrame(columns=final_columns)

    # Working on Each document 

    unique_files = concat_df['doc'].unique()
    
    results_sig_df= pd.DataFrame(columns=['doc_id','word', 'JSD']) 

    def ifnull(a, default_value):
        return a if a is not None else default_value

    for word  in unique_files:  #loop per cod
        filtered_df = concat_df[concat_df['word'] == word]  # df only for 1 cod = vector
        for index, row in filtered_df.iterrows():
            p=  row['wordavg']  #Constant All the COD's and years
            q = row['percent_of_total']  # list of All the COD's in specific year  
            klpart2=  row['wordavg']/row['percent_of_total']  
            kl_distance =  (row['wordavg'] -row['percent_of_total'] ) *   np.log10(klpart2)
            kl_distance =    np.log10(klpart2)

            H=row['percent_of_total']
            I=row['wordavg']
            J= (ifnull(H, 0) +  ifnull(I, 0)) /2
            K=np.log2(H/J)*H
            L=np.log2(I/J)*I
            M=0.5*(K+L)



            #js_distance = calcs.calculate_jsd_sig_distance(p, q)  # return 1 value per year per cod 
            #js_distance_general=calcs.calculate_js_distance(p,q)
            # append the results to the dataframe by location  
            new_row = pd.DataFrame({'doc_id': [word], 'JSD': [M],'word':[row['word']],  'H':[H],'I':[I],'J':[J],'K':[K],'L':[L],'M':[M] }) 
            print(new_row) 
            results_sig_df =  pd.concat([results_sig_df,new_row], ignore_index=True)
            np.multiply(np.where(results_sig_df['I'] < results_sig_df['H'], -1, 1), results_sig_df['JSD'], out= results_sig_df['JSD'])



        print (results_sig_df)    
    
        

    

    # write the results of signatures into CSV

    results_sig_df.to_csv(current_path+'/Signatures/results_'+alias+'/signatures_word_30.csv', index=True)  
    
    # Create distances matrix for frequencies 
   

    # distances Sockpuppet
    from scipy.spatial.distance import pdist, squareform 
    import numpy as np

    # input is original frequencies vector 

    summary = localdf.groupby(['doc_id','word'])['percent_of_total'].agg(['sum']).reset_index() 
    distances_sockpuppet.sockpuppet(summary,'Orig-euc','euclidean',alias) 
    distances_sockpuppet.sockpuppet(summary,'ORIG-JSD','jensenshannon',alias) 

    # input is JSD results vector 

    summary = results_sig_df.groupby(['doc_id','word'])['JSD'].agg(['sum']).reset_index() 
    distances_sockpuppet.sockpuppet(summary,'jensenshannon-JSD','jensenshannon',alias) 
    distances_sockpuppet.sockpuppet(summary,'jensenshannon-euc','euclidean',alias)   

#sig(alias)  