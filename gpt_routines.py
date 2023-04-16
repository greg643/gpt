"""
Required Libraries to Import
"""

#generic
import platform
import os

#core data wrangling
import pandas as pd
#show all columns in dataframe
pd.set_option('display.max_columns', None)

import json
import xmltodict
from bs4 import BeautifulSoup #as bs
import time
import markdown as md
from ast import literal_eval

#regular expressions
import re
import io

#html requests
import requests

#for pinecone
from tqdm.auto import tqdm

#chatgpt
import openai
from openai import APIError
from openai.embeddings_utils import get_embedding
import pinecone

#for saving files
import pickle

#for reading PDFs
from pdfminer.high_level import extract_text
#from PyPDF2 import PdfReader
from uuid import uuid4

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import tiktoken

"""
API Keys

Yes, I know. I trust you.
"""

#####    
##### OPEN AI & PINE CONE
#####

# openai.api_key = "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";
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

# pinecone API key
PINECONE_API_KEY = '6cdc78da-7ca2-4b6e-9b5f-ddaadfa753c7'
PINECONE_API_ENV = 'us-central1-gcp'


"""
Critical Helper Functions:
Counting Tokens & Chunking Text

Using "tiktoken" counter which is the most precise estimator for GPT-4
"""

#####
##### CHUNK INPUT TEXT
#####

def break_up_file_to_chunks(text, chunk_size=3500, overlap_size=100):
    tokens = word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))

def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)

def convert_to_detokenized_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text

#####
##### GET CORRECT TOKENS FOR GPT4
#####

def num_tokens_from_string(string: str, encoding_name ="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

"""
KEY FUNCTION #1:
CHUNK A DOCUMENT TO PINECONE

This function assumes you have already extracted some datasource to clean text.
It will use various helper functions to:
* Chunk your text into various sizes
* Call the OpenAI embeddings API
* Store the embeddings as vectors in Pinecone 

Because this is muggle-level stuff, I have a deus-ex-machina in here, in that:
* Using a free Pinecone database - everything in one database
* I try to avoid any helper libraries - eg, langchain - I'm trying to stay as close to base APIs as possible
* I just delete the vector DB every time this is called, so you're guaranteed to be looking at one document at a time.

There is some detritus in all of these functions, because I'm using these functions for a bunch of other projects.
Because I haven't fully abstracted all the details, I just work around these when I find them.
"""

def chunk_to_pinecone(df):
    
    df_return = pd.DataFrame(columns=['GUID', 'Name', 'Link', 'Tokens', 'Chunk Number', 'Chunk Text', 'Embeddings'])

    #THIS IS OUR PINECONE DATABASE
    index_name = "langchain3"
    
    # Initialize connection to Pinecone
    pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV
    )
    
    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index.describe_index_stats()
    
    # # Check if index already exists, create it if it doesn't
    # if index_name not in pinecone.list_indexes():
    #  pinecone.create_index(index_name, dimension=1536, metric='dotproduct')
    
    # clear out a pinecone database
    index.delete(deleteAll='true', namespace='')
    #row = df_raw.iloc[0:1]
    # PINECONE - CREATE VECTOR DATABASE
    #for i, row in df_raw.iloc[0:1].iterrows():
        
    text = str(df['Text'][0])     
    text_chunks = break_up_file_to_chunks(text, chunk_size = 1250)

    #for chunk in text_chunks:
    for i, chunk in enumerate(text_chunks):
    
        clean_text = convert_to_detokenized_text(chunk)
        
        embed_model = 'text-embedding-ada-002'

        response = openai.Embedding.create(
          input = clean_text,
          model = embed_model
        )
        
        embed = response['data'][0]['embedding']
    
        time.sleep(2)
            
        new_row = {'GUID': df['GUID'][0],
               'Name': df['Name'][0], 
               'Link': df['Link'][0], 
               'Tokens': len(chunk),
               'Chunk Number': i, 
               'Chunk Text': clean_text, 
               'Embeddings': embed,
               }
        
        new_df = pd.DataFrame([new_row])
            
        df_return  = pd.concat([df_return , new_df], axis=0, ignore_index=True)
               
    # Convert the DataFrame to a list of dictionaries
    # chunks = df_sources[(df_sources['Age'] == 22)].to_dict(orient='records')
    chunks =  df_return.to_dict(orient='records')
            
    for chunk in chunks:
        upsert_response = index.upsert(
            vectors=[
                {
                'id': chunk['GUID'] + str(chunk['Chunk Number']), 
                'values': chunk['Embeddings'], 
                'metadata':{
                    'Name': chunk['Name'],
                    'Link': chunk['Link'],
                    'Text': chunk['Chunk Text']
                }}
                
            ],
            namespace=''
        )
    print(index.describe_index_stats())


def ask(question):
    #question = questions['Tick Size Reform'][0]                
    # question = 'Does the text support access fee reform?'
    
    prompt_instr = " please limit your answer strictly to this question, and if the answer is unclear, please respond that the letter does not specifically address the question. "
              
    #THIS IS OUR PINECONE DATABASE
    index_name = "langchain3"
    
    # Initialize connection to Pinecone
    pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV
    )
    
    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index.describe_index_stats()
    
    user_input = question + prompt_instr
    
    embed_query = openai.Embedding.create(
        input=user_input,
        engine=embed_model
    )
    
    # retrieve from Pinecone
    query_embeds = embed_query['data'][0]['embedding']
    
    # get relevant contexts (including the questions)
    response = index.query(query_embeds, top_k=4, include_metadata=True)
    #, filter={"GUID": {"$eq": "1aee345a-82e0-4d70-88fc-15073c362c92"}} 
    
    contexts = [item['metadata']['Text'] for item in response['matches']]
    
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+user_input # + query
    
    # system message to assign role the model
    system_msg = f"""You are a helpul machine learning assistant and tutor. Answer questions based on the context provided, provide support for your answer, or say Unable to find reference."
    """
    
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        #model="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "assistant", "content": "\n".join(contexts)},
            {"role": "user", "content": question}
        
        ]
    )
    
    answer = chat['choices'][0]['message']['content'].strip()
    print(answer)
    return
  
"""
KEY FUNCTION #2:
ASK A QUESTION OF THE PINECONE DOCUMENT

This function assumes you have uploaded a vectorized document to Pinecone
Will simply ask one question at a time of that document, and return to the window.
"""
 


def ask(question):
    #question = questions['Tick Size Reform'][0]                
    # question = 'Does the text support access fee reform?'
    
    prompt_instr = " please limit your answer strictly to this question, and if the answer is unclear, please respond that the text does not specifically address the question."
              
    #THIS IS OUR PINECONE DATABASE
    index_name = "langchain3"
    
    # Initialize connection to Pinecone
    pinecone.init(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_API_ENV
    )
    
    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index.describe_index_stats()
    
    user_input = question + prompt_instr
    
    embed_query = openai.Embedding.create(
        input=user_input,
        engine=embed_model
    )
    
    # retrieve from Pinecone
    query_embeds = embed_query['data'][0]['embedding']
    
    # get relevant contexts (including the questions)
    response = index.query(query_embeds, top_k=4, include_metadata=True)
    #, filter={"GUID": {"$eq": "1aee345a-82e0-4d70-88fc-15073c362c92"}} 
    
    contexts = [item['metadata']['Text'] for item in response['matches']]
    
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+user_input # + query
    
    # system message to assign role the model
    system_msg = f"""You are a helpul machine learning assistant and tutor. Answer questions based on the context provided, provide support for your answer, or say Unable to find reference."
    """
    
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        #model="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "assistant", "content": "\n".join(contexts)},
            {"role": "user", "content": question}
        
        ]
    )
    
    answer = chat['choices'][0]['message']['content'].strip()
    print(answer)
    return
    
