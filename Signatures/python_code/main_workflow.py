import pandas as pd
import glob
import calcs   as calcs
import os
import text_preprocessing_spacy as text_preprocessing_spacy
import sig_after_preprocessing as sig_after_preprocessing
import distances_sockpuppet as distances_sockpuppet
import cod_vector_preprocessing as cod_vector_preprocessing
import sig_after_preprocessing_cod as sig_after_preprocessing_cod

dataset='cod'
current_path = os.getcwd()

if dataset=='cod':
    cod_vector_preprocessing
    sig_after_preprocessing_cod
    distances_sockpuppet
    print (dataset)