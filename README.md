# gpt
fun with gpt4 &amp; pinecone

GPT is probably the perfect marriage of Marc Andreesen's "wordcels" and "shape rotators". Every shape rotator can now be a wordcel, and every wordcel can learn to be shape rotator, although we know better - each tribe will just wordcel and shape rotate even harder.

The capability I have been most interested in for GPT is what I call the "verbal chainsaw," something that can help me make sense of an almost baffling amount of regulatory rules, academic papers and various rules and filings that I need to read to understand aspects of my work.

Although I was originally trained as a wordcel, with age and new skills, I have lost an enormous amount of patience for reading dry and boring material, although I have an almost endless amount of time and energy to read summaries and digests, and to dig in further when I have the right context.

So - in order to unlock these capabilities, I wanted to see if I could create a custom solution do do exactly this. I soon learned that dumping large texts into ChatGPT caps out pretty quickly, not to mention the tedium of copying data in - there has to be a better way.

And there is! 

Everyone is very excited about ChatGPT and rightly so, but with token limits being what they are, the transformational capability becomes what is called a "vector database." A vector database is a way to store "embeddings" or vector representations of specific documents, particularly very large documents that don't fit into a single prompt. When the user asks a question of a very large document, the vector database is employed to provide a nearest-neighbor semantic search, returning the "k" number of items that appear to answer the question. These prompts are then fed to GPT along with a specific question for an answer.

Prerequisites - you will need most of these, might as well install them all:

```
pip install pdfminer.six requests pytesseract openai requests nltk bs4 xmltodict pinecone-client tiktoken pdf2image markdown plotly
```
