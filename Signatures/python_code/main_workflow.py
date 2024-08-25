import calcs   as calcs
import os
import text_preprocessing_spacy_func as text_preprocessing_spacy_func
#import sig_after_preprocessing as sig_after_preprocessing
#import distances_sockpuppet as distances_sockpuppet
import vector_preprocessing_func as vp
import sig_after_preprocessing_func as sigs
import sig_after_preprocessing_cod_func as codsigs

dataset='cod'
current_path = os.getcwd()  

if dataset=='cod':
    vp.vector_preprocessing(current_path+'/Signatures/raw_data/raw_cod.csv') 
    sigs.sig(dataset) 
    #codsigs.sig(dataset) 


if dataset=='loco':  
    text_preprocessing_spacy_func.preprocessing(current_path+'/Signatures/raw_data/raw_'+dataset+'.csv') 
    vp.vector_preprocessing(current_path+'/Signatures/results_'+dataset+'/agg_merged.csv')
    sigs.sig(dataset) 
  