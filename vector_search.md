Update 4/17/23

## How does GPT "see" language?

GPT doesn't "see" language, it has text, and a vector representation of text. To make sense of the numbers, you need to pair the matrix with their corresponding text.

Interestingly, GPT does not directly interpret the meaning of the numbers, the models "learn" to represent text as a set of numerical values based on the patterns and relationships found in large amounts of text data. It processes text as a sequence of numbers, and uses statistical patterns in those sequences to make predictions about what text is likely to come next. 

This process is known as "training" the model, where the model is shown many examples of text and corresponding labels, and it learns to associate certain patterns in the input with certain outputs.

During inference, when the model is given a sequence of text, it generates a corresponding sequence of numbers that represents the text, and then uses that sequence to make predictions about what text should come next. It repeats this process for each step of the sequence, generating a new sequence of numbers at each step, and using those numbers to update its predictions.

## What does an embedding look like?

Raw text:

"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum." 

Embedding representation:

For GPT, this becomes a matrix of 1 deep x 768 dimensions:

...
-0.03910874202847481,
 -3.929028480342822e-07,
 -0.01148751936852932,
 -0.004625982139259577,
 -0.004807789344340563,
 0.03409894183278084,
 0.0119723379611969,
 -0.0006367459427565336,
 -0.00458558090031147,
 -0.002326458226889372,
 ...
 
## How are embeddings calculated?

A technically detailed yet hopefully still clear explanation: an embedding is a fixed-length vector representation of a word. It is generated roughly by "multiplying the one-hot encoded word vector with a weight matrix."

1. One-hot encoding: First, the word is "one-hot" encoded into a sparse vector. A sparse vector is the name for one that is almost all zeros; in this case, it's all zeros, except for the element corresponding with the word, which becomes a 1. Example, if the word is "apple" and it has an index of 50 in the vocabulary, then the one-hot encoded vector would have a value of 1 at index 50 and 0s everywhere else.

2. Weight matrix multiplication: the "one-hot" encoded vector is multiplied bya weight matrix W, which has dimensions of (vocab_size x embedding_size). The result of this multiplication is a dense vector of size (1 x embedding_size). Each element in the resulting dense vector is a weighted sum of the rows in the weight matrix corresponding to the non-zero elements in the one-hot encoded vector. This dense vector is the final embedding for the word.

3. Non-linearity: In some cases, a non-linear activation function such as ReLU or tanh is applied element-wise to the dense vector to introduce non-linearity into the model.

The weight matrix W is initially randomly initialized for the LLM and is learned during training through backpropagation, which updates the weights to minimize the loss function that measures the difference between the predicted output of the model and the true output. Once the training is completed, the learned weight matrix is used to generate embeddings for new words or tokens.

## How does vector search work?

Similar words are determined to be "close together" in the embedding space, and dissimilar words are "far apart", mathematically speaking.

Closeness is measured by the angle of the vectors, such as with "cosine distance". The cosine distance is calculated by taking the cosine of the angle between the two vectors. If the vectors are very similar, the cosine distance will be small, and if they are very different, the cosine distance will be large.

The term "angle" sort of implies a geometric concept, like we could almost visualize this, but we are comparing high-dimensional complex vectors, and the visualization becomes more difficult as the number of dimensions increases. Nonetheless, the mathematical concept remains the same.
