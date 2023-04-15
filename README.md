#GPT & Vector Databases
##or, fun with GPT4 &amp; pinecone

GPT is probably the perfect marriage of Marc Andreesen's "wordcels" and "shape rotators". Every shape rotator can now be a wordcel, and every wordcel can learn to be shape rotator, although we know better - folks will just use GPT to wordcel and shape rotate even harder.

The capability I have been most interested in for GPT is what I call the "verbal chainsaw," something that can help me make sense of an almost baffling amount of regulatory rules, academic papers and various rules and filings that I need to read to understand aspects of my work.

Although I was originally trained as a wordcel, with age and new skills, I have lost an enormous amount of patience for reading dry and boring material, although I have an almost endless amount of time and energy to read summaries and digests, and to dig in further when I have the right context.

So - in order to unlock these capabilities, I wanted to see if I could create a solution for this. I soon learned that dumping large texts into ChatGPT caps out pretty quickly, not to mention the tedium of copying data in - there has to be a better way.

And there is! 

Everyone is very excited about ChatGPT and rightly so, but with token limits being what they are, the transformational capability becomes what is called a "vector database." 

A vector database is a way to store "embeddings" or vector representations of specific documents, particularly very large documents that are too big for a single GPT prompt. 

When the user asks a question of a very large document, the vector database is employed to provide a semantic search, returning the "k" number of items that appear to answer the question. These semantic matches are then fed to GPT along with a specific question for an answer.

The magic of this particular solution is being able to marry the sophistication and larger token limits of GPT-4 with the Pinecone vector database.

Keep in mind that this is all muggle stuff - very simple routines, so that we can precisely understand what is happening with our document and code. Eventually,this process can scale!

For Python, you will need most of these, might as well install them all:

```
pip install pdfminer.six requests pytesseract openai requests nltk bs4 xmltodict pinecone-client tiktoken pdf2image markdown plotly
```
The core routine is below:

The core libraries are:

