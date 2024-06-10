import pandas as pd
import tiktoken
from openai import OpenAI
import os
import numpy as np
from embeddingUtils import distances_from_embeddings
# import pandas as pd
# from openai import OpenAI
# import os

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Create a list to store the text files
def create_text_files_frame(domain, text_dir="text/", url_dir="urls/", process_dir="processed/"):
    texts=[]

    if not os.path.exists(process_dir +"/"):
        os.mkdir(process_dir)
    # Get all the text files in the text directory
    for file in os.listdir(text_dir + domain + "/"):

        # Open the file and read the text
        url = "https://" + domain + "/"
        with open(url_dir + domain + "/" + file, "r") as f:
            url = f.read()

        with open(text_dir + domain + "/" + file, "r") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((url, file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['url', 'fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(process_dir+'scraped.csv')
    df.head()

    df = pd.read_csv(process_dir+'scraped.csv', index_col=0)
    df.columns = ['url', 'title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df

# Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, url, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append((url,". ".join(chunk) + "."))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append((url,". ".join(chunk) + "."))

    return chunks

def build_embeddings(domain, text_dir="text/", url_dir="urls/", process_dir="processed/"):
    df = create_text_files_frame(domain, text_dir, url_dir, process_dir)

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        url = row[1]['url']
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], url)

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append((url, row[1]['text'] ))

    df = pd.DataFrame(shortened, columns = ['url', 'text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # df.n_tokens.hist()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)

    df.to_csv(process_dir+'embeddings.csv')
    df.head()
