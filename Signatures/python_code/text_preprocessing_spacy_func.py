import spacy
import re
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from nltk.stem import PorterStemmer, LancasterStemmer
import os
#import Signatures._LPA
#import Signatures._TLPA
# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the English pipeline
nlp = spacy.load("en_core_web_sm")

# Initialize stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer() 
 
def text_preproc(alias): 
    # Get the current working directory
    current_path = os.getcwd()
    print(current_path)

    #Modify the input file name as required 
    filename = current_path+'/Signatures/raw_data/raw_'+alias+'.csv'

    # Download required resources
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    # Load the English pipeline
    nlp = spacy.load("en_core_web_sm")

    # Initialize stemmers
    porter = PorterStemmer()
    lancaster = LancasterStemmer() 

    # Get the current working directory
    current_path = os.getcwd()
    print(current_path)
 
    df = pd.read_csv(filename)
     
    # Read the CSV file
    return (df)




def tokenize(text):
    """Tokenize the given text."""
    doc = nlp(text)
    return [token.text for token in doc]

def lemmatize(text):
    """Lemmatize the given text."""
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def classify(text):
    """Classify named entities in the given text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def lemmatize_and_tag(text):
    """Lemmatize the given text and tag if it's a noun or adjective."""
    doc = nlp(text)
    return [(token.lemma_, token.pos_) for token in doc if token.pos_ in ['NOUN', 'ADJ']]  
    
# Same code with NLTK 

def get_wordnet_pos(treebank_tag):
    """Map treebank POS tag to first character used by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatize_and_tag_nltk(text):
    """Lemmatize the given text and tag if it's a noun or adjective using nltk."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    lemmatizer = WordNetLemmatizer()

    return [(lemmatizer.lemmatize(token, get_wordnet_pos(pos)), pos) for token, pos in pos_tags    if pos in ['NN','NNS', 'JJ'] or pos.startswith("V") ]


# Stemming with Porter Stemmer
def stem_and_tag_nltk(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    return  [(porter.stem(token, get_wordnet_pos(pos)), pos) for token, pos in pos_tags    if pos in ['NN','NNS', 'JJ'] or pos.startswith("V") ] 

# Stemming with Lancaster Stemmer
# lancaster_stems = [lancaster.stem(word) for word in words] 


def preprocessing(input_file): 
    
    #get the file alias from file_name
    match = re.search(r'(?<=raw_)\w+(?=\.csv)', input_file)
    if match:
        print(match.group())  # Output: cod
        alias=match.group()
    else:
        print("No match found")

    
    df= text_preproc(alias)  
    print (df.head(10))
    if alias == 'loco' :
        df['vector']=df['doc_id']
    result_spacy_csv = []
    result_nltk_csv  = []
    result_stem_csv  = []     
    
    # Process each text in the CSV - spacy

    for index, row in df.iterrows(): 
        text = row['txt']
        doc_id=row['doc_id'] 
        dt=row['dt'] 
        subcorpus=row['subcorpus']
        if text is None or pd.isnull(text):
            continue 
        #Clean spcial characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Uncomment one of these 3 lines 

        results = lemmatize_and_tag(text)
        #results_nltk=lemmatize_and_tag_nltk(text)
        #results_stem=stem_and_tag_nltk(text)


        #print(f"Text {index + 1}:")
        
        for lemma, pos in results:
            print(f"Spacy Lemmatized: {lemma}, POS: {pos}")
            result_spacy_csv.append({"word":lemma,"pos":pos,"vector":index + 1 , "doc_id": doc_id , "subcorpus":subcorpus, "dt":dt})
        print("\n")
        """ 
        for lemma, pos in results_nltk:
            print(f"nltk Lemmatized: {lemma}, POS: {pos}")
            result_nltk_csv.append({"word":lemma,"pos":pos,"vector":index + 1}) 
        print("\n")
        
        for stem, pos in results_stem:
            print(f"nltk stem: {lemma}, POS: {pos}")
            result_stem_csv.append({"word":stem,"pos":pos,"vector":index + 1}) 
        print("\n") 
        """
    current_path = os.getcwd() 
    # Convert the result to a DataFrame and save it to a CSV
    df = pd.DataFrame(result_spacy_csv) #( result_stem_csv)
    df.to_csv(current_path+'/Signatures/results_'+alias+'/result_spacy_csv.csv', index=False)
    df['doc']=df['vector'] 
    # Group by 'word' column and count (DVR)

    result = df.groupby('word').agg({
        'pos': 'max',           # max of 'Value' column
        'doc_id':'max',
        'doc':'max',
        'dt':'max',
        'word': 'count',
        'subcorpus':'max'        # count of 'OtherColumn'
    }).rename(columns={'pos': 'Maxpos', 'word': 'wordCount', 'doc':'doc'})


    dfglobal=result

    
    # Generate vectors per author 
    dflocal = df.groupby(['vector', 'word']).agg({
        'pos': 'max',           # max of 'Value' column
        'word': 'count',         # count of 'OtherColumn' 
        'dt':'max', 
        'doc_id':'max',
        'doc':'max',
        'subcorpus':'max'        # count of 'OtherColumn'
    }).rename(columns={'pos': 'Pos', 'word': 'wordCount','doc': 'doc','doc_id':'doc_id'  }) 

    #print(dflocal) 
    #dflocal.to_csv('agg_local.csv', index=True)

    # Calculate Frequencies and distances 

    # Calculate the total sum
    total_sum = dfglobal['wordCount'].sum()
    print(total_sum)
    # Calculate the percentage of the total for each value (local)
    
    total_sum_by_category = dflocal.groupby('vector')['wordCount'].transform('sum')

    # Calculate the percent of total within each category
    dflocal['percent_of_total'] = (dflocal['wordCount'] / total_sum_by_category)  
    # Optionally, sort the DataFrame by 'category' and 'percent_of_total'
    #dflocal = dflocal.sort_values(['vector', 'percent_of_total'], ascending=[True])


    all_local_freq_file_name= current_path+'/Signatures/results_'+alias+'/all_local_freq.csv'
    dflocal.to_csv(all_local_freq_file_name, index=True)

    # Split to multiple files   

    grouped = dflocal.groupby(level=0)
    # Iterate over each group
    for vector_value, group_df in grouped:
        # Define the CSV file name based on the vector value
        file_name = current_path+'/Signatures/results_'+alias+'/'+f'data_{vector_value}.csv'
        
        # Save the group DataFrame to a CSV file
        group_df.to_csv(file_name)
        
        #print(f"DataFrame for vector {vector_value} saved to {file_name}")
    print(vector_value)
    
    # Calculate global Frequencies (avg of avg) 
    dfglobal = dflocal.groupby(['word']).agg({
        'Pos': 'max',           # max of 'Value' column
        'percent_of_total': 'mean' ,         # avg of avgs  
        'wordCount':'sum',  
    }).rename(columns={'Pos': 'Pos', 'percent_of_total': 'avg_freq' , 'wordCount':'wc'    }) 

    dfglobal['prevalence']=dfglobal['wc']/vector_value

    # calculate avg of avg for a specific word 


    dfglobal['wordavg']=dfglobal['wc']/total_sum 
    dfglobal['global_freq'] = dfglobal['avg_freq']

    all_global_freq_file_name=current_path+'/Signatures/results_'+alias+'/all_global_freq.csv'
    dfglobal.to_csv(all_global_freq_file_name, index=True)
    
    # Merge dflocal and dfglobal to the same CSV file  
    merged_df = pd.merge(dflocal,dfglobal, on='word',   how='left')
    if alias=='loco':
        merged_df['vector']=merged_df['doc']
    #print(merged_df) 
    merged_df.to_csv(current_path+'/Signatures/results_'+alias+'/agg_merged.csv', index=True) 


def calc_frequencies(input_file):
    import re

    filename = input_file  
    df = pd.read_csv(filename)
    df['vector']=df['doc_id']
    # Get the current working directory
    current_path = os.getcwd()
    print(current_path) 
    
    #get the file alias from file_name
    match = re.search(r'(?<=raw_)\w+(?=\.csv)', input_file)
    if match:
        print(match.group())  # Output: cod
        alias=match.group()
    else:
        print("No match found")

    print('alias:' ,alias)    



    result = df.groupby('word').agg({
        'doc_id':'max',
        'doc':'max',
        'dt':'max',
        'wordCount': 'sum'         
    }).rename(columns={ 'wordCount': 'wordCount', 'doc':'doc'})


    dfglobal=result

    
    # Generate vectors per author 
    dflocal = df.groupby(['vector', 'word']).agg({ 
        'word': 'max',         # count of 'OtherColumn' 
        'dt':'max', 
        'doc_id':'max',
        'doc':'max' , 
        'wordCount': 'sum'   
    }).rename(columns={ 'wordCount': 'wordCount','doc': 'doc','doc_id':'doc_id'  }) 

    #print(dflocal) 
    #dflocal.to_csv('agg_local.csv', index=True)

    # Calculate Frequencies and distances 

    # Calculate the total sum
    total_sum = dfglobal['wordCount'].sum()
    print(total_sum)
    # Calculate the percentage of the total for each value (local)
   
    dflocal['total_sum_by_category'] = dflocal.groupby('vector')['wordCount'].transform('sum')

    # Calculate the percent of total within each category
    dflocal['percent_of_total'] = (dflocal['wordCount'] / dflocal['total_sum_by_category'])  
    # Optionally, sort the DataFrame by 'category' and 'percent_of_total'
    dflocal = dflocal.sort_values(['vector'], ascending=[True])


    all_local_freq_file_name= current_path+'/Signatures/results_'+alias+'/all_local_freq.csv'
    dflocal.to_csv(all_local_freq_file_name, index=True)

    # Split to multiple files   

    grouped = dflocal.groupby(level=0)
    # Iterate over each group
    for vector_value, group_df in grouped:
        # Define the CSV file name based on the vector value
        file_name = current_path+'/Signatures/results_'+alias+'/'+f'data_{vector_value}.csv'
        
        # Save the group DataFrame to a CSV file
        group_df.to_csv(file_name)
        
        #print(f"DataFrame for vector {vector_value} saved to {file_name}")
    print(vector_value)
    
    # Calculate global Frequencies (avg of avg) 
    dfglobal = dflocal.groupby(['word']).agg({
        # max of 'Value' column
        'percent_of_total': 'mean' ,         # avg of avgs  
        'wordCount':'sum',  
    }).rename(columns={ 'percent_of_total': 'avg_freq' , 'wordCount':'wc'    }) 

    dfglobal['prevalence']=dfglobal['wc']/vector_value

    # calculate avg of avg for a specific word 


    dfglobal['wordavg']=dfglobal['wc']/total_sum 
    dfglobal['global_freq'] = dfglobal['avg_freq']

    all_global_freq_file_name=current_path+'/Signatures/results_'+alias+'/all_global_freq.csv'
    dfglobal.to_csv(all_global_freq_file_name, index=True)
    
    # Merge dflocal and dfglobal to the same CSV file  
    merged_df = pd.merge(dflocal,dfglobal, on='word',   how='left')
    if alias=='loco' or alias=='codcopy':
        merged_df['vector']=merged_df['doc']
    #print(merged_df) 
    merged_df.to_csv(current_path+'/Signatures/results_'+alias+'/agg_merged.csv', index=True) 



