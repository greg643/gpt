## GPT & Vector Databases
### (or, fun with GPT4 &amp; pinecone)


GPT is probably the perfect synthesis of Marc Andreesen's "wordcels" and "shape rotators". Every shape rotator can now have the skills of a master wordcel, and every wordcel can learn to be shape rotator! 

<blockquote class="twitter-tweet" data-width="550" data-lang="en" data-dnt="true" data-theme="light"><p lang="en" dir="ltr">Why do wordcels win head to head fights with shape rotators? Shape rotators spend 90% of their time rotating shapes and only 10% wordcelling; wordcels wordcel 24x7. Asymmetric warfare, outcome predetermined.</p>&mdash; Marc Andreessen (@pmarca) <a href="https://twitter.com/pmarca/status/1488985078545874944">Feb 2, 2022</a></blockquote>

## The Verbal Chainsaw

The capability I have been most interested in for GPT is what I call the "verbal chainsaw," something that can help me make sense of an almost baffling amount of regulatory rules, academic papers and various rules and filings that I feel I need to read to understand my field.

The goal hers is that with the verbal chainsaw, we can more confidently assess a specific domain, and extract specific insights from folks who have done deep and careful work, to advance our understanding.

Using the ChatGPT web interface, I learned immediately that dumping large texts into ChatGPT caps out quickly, not to mention the tedium of copying data in - there has to be a better way.

Inspired by a number of projects I saw online, I wanted to create something that I could use on-demand, for my exact objectives, without the limits imposed by a third party tool (interfaces, outputs, gpt models, etc). 

## Enter the Vector Database

With token limits being what they are, I soon realized that the transformational capability of an LLM can't really be unlocked without the ability to query over specific data, which requires something called a "vector database." 

A vector database is a way to store "embeddings," or vector representations of specific documents, particularly very large documents that are too big for a single GPT prompt, or even collections of documents, such as an entire confluence site.

When the user asks a question of a very large document, the vector database is employed to provide a semantic search, returning a set number of items that appear to answer the question. These "semantic matches" are the precursor to an answer, as blocks of text that might have the answer, then are then fed to GPT along with a specific question for an answer. GPT does the hard work of creating meaning from the matches, and since it only processes the excerpts, we work around token limits.

The magic here, is that the the vector database acts as a sort of missing memory for GPT, which is essential when you don't want it to just free associate an answer, but when you want a specific answer from a piece of information, and to return back for additional insights.

This particular solution is made more powerful by use of GPT-4, which allows access to the best GPT engine with higher token limits, along with the memory veatures of the Pinecone vector database. However, it also works with GPT3.5, but need to adjust token limits accordingly.

Keep in mind that this is all muggle level stuff - simple routines, so we can understand what is happening with our document and code. Once the use case is working well, this process can scale!

## Special Note

Since I originally created this, I have learned that the vector database is not strictly needed - apparently, I can do the semantic search on a single document from the embeddings I've already created and have in memory. 

This approach is somewhat overcomplicated because I adapted it from people who were creating custom document databases, which led me down the road of using a vector database. 

OpenAI's recently published example:

https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

## Prerequisites

For Python, you will need most of these, might as well install them all:

```python
pip install pdfminer.six requests pytesseract openai requests nltk bs4 xmltodict pinecone-client tiktoken pdf2image markdown plotly
```

## Core Python Routine

The core python routine to extract from a document is below. I'm including a PDF that already has a text layer, so no OCR is required.

For this to work, you will also need to import all of the libraries and functions found in: https://github.com/greg643/gpt/blob/main/gpt_functions.py

```python
####
#### EXTRACT FROM A WEB LINK
####

url = 'https://www.sec.gov/comments/s7-30-22/s73022-20160364-328968.pdf'

r = requests.get(url, headers=headers, allow_redirects=True)

text = extract_text(io.BytesIO(r.content))

text = clean_extracted_text(text)

####
#### EXTRACT FROM A SAVED PDF
####

file_name = '/Users/greg/Dropbox/_Industry Papers/IEX_Comment_Letter_s73022-20160364-328968.pdf'

text = extract_text(file_name)

# Verify we have what we want
print(text)

#simple function to clean pdf extracts
text = clean_extracted_text(text)

####
#### UPLOAD YOUR DOCUMENT TO PINECONE
####

# Each Vector will have a unqiue ID - for now, we are using python uuids + chunk numbers
# For various reasons, I am handling everything as dataframes.

uuid = str(uuid4())
new_row = {
            'GUID': uuid,
            'Name': file_name, 
            'Link': 'Link', ##detritus, leave alone
            'Tokens': num_tokens_from_string(text), 
            'Text': text,
            }

new_df = pd.DataFrame([new_row])

chunk_to_pinecone(new_df)

####
#### ASK QUESTIONS
####

question = 'what does this letter say about the proposed structure for retail auctions?'

ask(question)
```

Obviously this could be coded into a webite etc - for now you can just enter questions into python and run over and over.


