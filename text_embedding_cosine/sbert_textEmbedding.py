from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import numpy as np

# load model SBERT 
model = SentenceTransformer('all-MiniLM-L6-v2')  

# example of sentences 
sentence1 = 'I will learn some the artificial intelligence topics like machine learning, deep learning, LLM, and many more'
sentence2= "I want to study machine learning and artificial intelligence"

''' 
Here below another sentences for sentences2, so you can compare between these two sentences:

'the ai lessons that i need to learn is machine learning, GenAI, deep learning, LLM, Data science, etc.'1
'i love cooking fried rice and fried noodle'
'''

# to get the embedding vector
vec1 =  model.encode(sentence1)
vec2 =  model.encode(sentence2)  

# to know the similarity between 2 embedding (using Cosine similarity)
similarity = cosine_similarity([vec1], [vec2]) 


# check the shape and the vector 
print('\nHere below is the Vec1 result\n')  
# print('Vec1 result: ', vec1)
print('vec 1 shape: ', vec1.shape)  
print('vec 1 (first 10 number): ', vec1[:10])

print('================================================================================')  

print('\nHere below is the Vec2 result\n')  
# print('Vec2 result: ', vec2)
print('vec 2 shape: ', vec2.shape)  
print('vec 2 (first 10 number): ', vec2[:10])

# print('\n================================================================================\n')  
print('Cosine similarity:', similarity[0][0])
print()