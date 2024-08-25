# DHTA
Dynamic High-dimensional Timeline Analysis (DHTA)

# For Preprocessing install 

צריך להתקין את הספריות הבאות 

pip install spacy
python -m spacy download en_core_web_sm
pip install nltk
pip install pandas
pip install pingouin
pip install pyLDAvis
pip install wordcloud

# For LDA,Signatures install 
pip install gensim  
pip install string 
pip install re 
pip install wordcloud  
pip install matplotlib 
pip install numpy  
pip install random 


Run pip install -r requirements.txt. Make sure your Python version is up to date.

# 1. Run Preprocessing 

/PreProcessing/text_preprocessing_spacy.py

Results are writen into /PreProcessing/results  directory 

# 2. Run LDA

# 3. Run Signatures

Installation
Run pip install -r requirements.txt. Make sure your Python version is up to date.


Input :
We need to provide the dataset name as a parameter , and locate the raw data csv in the /Signatures/raw_data directory .
For "cod" dataset we will have a file /Signatures/raw_data/raw_cod.csv 

raw data sample : 

word,           wordCount,  dt,         doc_id,  doc 
Acute_hepatitis, 166343,    01/01/1990, 1990,    1990 
Alcohol_use,     116390,    01/01/1990, 1990,    1990 
Alzheimer,       560616,    01/01/1990, 1990,    1990 

We can also provude a pre-processed vector (of distributions)  in the following format : 

vector,word,wordCount,dt,doc_id,doc,percent_of_total 
1990,Acute_hepatitis,166343,01/01/1990,1990,1990,0.00382235 
1990,Alcohol_use_disorders,116390,01/01/1990,1990,1990,0.002674494 
1990,Alzheimer_s_disease_and_other_dementias,560616,01/01/1990,1990,1990,0.012882241 
1990,Cardiovascular_diseases,12062179,01/01/1990,1990,1990,0.277173491 
