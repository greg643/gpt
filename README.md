## GPT & Vector Databases
### (or, fun with GPT4 &amp; pinecone)


GPT is probably the perfect synthesis of Marc Andreesen's "wordcels" and "shape rotators". Every shape rotator can now have the skills of a master wordcel, and every wordcel can learn to be shape rotator! 

<blockquote class="twitter-tweet" data-width="550" data-lang="en" data-dnt="true" data-theme="light"><p lang="en" dir="ltr">Why do wordcels win head to head fights with shape rotators? Shape rotators spend 90% of their time rotating shapes and only 10% wordcelling; wordcels wordcel 24x7. Asymmetric warfare, outcome predetermined.</p>&mdash; Marc Andreessen (@pmarca) <a href="https://twitter.com/pmarca/status/1488985078545874944">Feb 2, 2022</a></blockquote>

(Although - more than likely - each tribe will just use GPT to wordcel and shape rotate even harder...)

## The Verbal Chainsaw

The capability I have been most interested in for GPT is what I call the "verbal chainsaw," something that can help me make sense of an almost baffling amount of regulatory rules, academic papers and various rules and filings that I need to read to understand aspects of my work.

I think my fundamental belief is that while we can't boil down the ocean, with the verbal chainsaw, we can probably now boil down large lakes, to extract insights from specific domains and help advance our understanding.

Although I was originally trained as a wordcel, with age and new skills, and maybe just age, I have lost an enormous amount of patience for reading dry material, although I have an almost endless amount of time and energy to read.

So - in order to unlock these capabilities, I wanted to see if I could create a solution for this. I soon learned that dumping large texts into ChatGPT caps out pretty quickly, not to mention the tedium of copying data in - there has to be a better way.

And there is! 

## Enter the Vector Database

Everyone is very excited about ChatGPT and rightly so, but with token limits being what they are, the transformational capability becomes what is called a "vector database." 

A vector database is a way to store "embeddings" or vector representations of specific documents, particularly very large documents that are too big for a single GPT prompt. 

When the user asks a question of a very large document, the vector database is employed to provide a semantic search, returning "k" number of items that appear to answer the question. These semantic matches are then fed to GPT along with a specific question for an answer.

In essence, the vector database acts as the missing memory for GPT, when you want it to rember and return to a specific piece of material.

This particular solution is made further powerful by use of GPT-4, which allows access to the best GPT engine with higher token limits, along with the memory veatures of the Pinecone vector database.

Keep in mind that this is all muggle level stuff - simple routines, so we can understand what is happening with our document and code. Eventually,this process can scale!

## Prerequisites

For Python, you will need most of these, might as well install them all:

```
pip install pdfminer.six requests pytesseract openai requests nltk bs4 xmltodict pinecone-client tiktoken pdf2image markdown plotly
```
You will also need to import all of the libraries and functions stored in https://github.com/greg643/gpt/blob/main/gpt_routines.py

## Core Python Routine

The core python routine is below:

```
####
#### DOCUMENT EXTRACTOR
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
            'Link': 'Link', 
            'Tokens': num_tokens_from_string(text), #, "cl100k_base"
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


