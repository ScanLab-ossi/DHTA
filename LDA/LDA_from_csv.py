import pandas as pd
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import spacy
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import random
# Read text data from CSV file
# Assume that the CSV file has a column named 'text'
# Get the current working directory
current_path = os.getcwd()
print(current_path)

# Read text data from two CSV files
# Assume both CSV files have a column named 'text'
csv_file_path1 = current_path+'/PreProcessing/raw_data/csv/WordFrequencies_data_human.csv' 
csv_file_path2 = current_path+'/PreProcessing/raw_data/csv/WordFrequencies_data_empty.csv' 
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)

# Concatenate the DataFrames
df = pd.concat([df1, df2]).reset_index(drop=True)
  

# Pre-processing the texts
def preprocess(document):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(document.lower())
    words_cleaned = [word for word in words if word not in stop_words and word not in string.punctuation]
    return words_cleaned

# Tokenize and clean text

df1=df['Team 1']
df2=df['Team 2']
df3=df['Team 3']
df4=df['Team 4']
df5=df['Team 5']
result_csv=[]
s = pd.concat([df1, df2, df3,df4,df5], ignore_index=True)
df = s.to_frame(name="txt")
for index, row in df.iterrows(): 
    text = row['txt']
    if text is None or pd.isnull(text):
        continue 
    #Clean spcial characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    if text is None or pd.isnull(text):
        result_csv.append({"text":"NULL","index":index + 1})
    else:
        result_csv.append({"text":text,"index":index + 1}) 

df = pd.DataFrame(result_csv)
print(df)


 
texts = [preprocess(document) for document in df['text']]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Apply LDA
# The passes parameter controls how often we train the model on the entire corpus
NUM_TOPICS = 6
lda_model = gensim.models.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from ipywidgets import interactive
import matplotlib.pyplot as plt
# Prepare the visualization data
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# Visualize the topics
pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, current_path+'/LDA/results/lda_visualization_human.html')

# wordcloud 
def my_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(220, 100%%, %d%%)" % random.randint(40, 100)

 

# Generate WordCloud for each topic
for t in range(lda_model.num_topics):
    plt.figure()
    
    # Prepare data
    terms = lda_model.show_topic(t, 20)
    term_dict = {}
    for term, freq in terms:
        term_dict[term] = np.round(freq, 4)
    
    # Generate WordCloud
    wordcloud = WordCloud(background_color='white',
                          width=800,
                          height=800,
                          color_func=my_color_func,
                         ).generate_from_frequencies(term_dict)
    
    # Display WordCloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Topic #{t+1}')
    nm=current_path+'/LDA/results/'+'wordcloud'+str(t+1)+'.png' 
    plt.savefig( nm, format='png')
    plt.show()


topic_data = []

# Print topics and words associated with it
topics = lda_model.print_topics(num_words=8)
for topic in topics:
    print('***',topic)
    topic_id, words_probs = topic
    words_probs = words_probs.split(" + ")
    words_probs = {word.split("*")[1].strip('"'): float(word.split("*")[0]) for word in words_probs}
    topic_data.append({'TopicID': topic_id, **words_probs})

# Convert to DataFrame
topics_df = pd.DataFrame(topic_data) 
# If you want to reset the index for better readability
topics_df.reset_index(inplace=True)

print(topics_df)

topics_df.to_csv(current_path+'/LDA/results/topics_h.csv', index=True) 
# Pivot the DataFrame to have TopicID as columns and words as index
pivoted_topics_df = topics_df.melt(id_vars=['TopicID'], var_name='Word', value_name='Probability')
pivoted_topics_df = pivoted_topics_df.pivot(index='Word', columns='TopicID', values='Probability')

# If you want to reset the index for better readability
pivoted_topics_df.reset_index(inplace=True)

print(pivoted_topics_df)

pivoted_topics_df.to_csv(current_path+'/LDA/results/topics_h_pivotted.csv', index=True)
