# Multiple sentences + Clustering 

from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from openai import OpenAI
import requests
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

# example of sentences 
''' list_of_sentences = [
    'i love deep learning',          # Tech
    'i like traveling to new york',  # Traveliing 
    'machine learning is awesome',    # tech
    'i enjoy cooking asian foods',   # food 
    'lets train some model of neural network',  # tech
    'i wanna go to singapore for vacation', # travelling 
    'cooking fried rice is delicious and i like it'  # food 
]'''

n_sentences = int(input('\nhow many sentences u wanna input?\n'))  # total = 7 sentences 
print()
list_of_sentences = []

for i in range(n_sentences):
    sentences_input = input()
    list_of_sentences.append(sentences_input)
# print(list_of_sentences)

embeddings = model.encode(list_of_sentences)

# how many clusters?  definisikan ada berapa banyak cluster/kelompok/grup 
cluster_num= int(input('\nhow many cluster? \n'))    # total  =  3 clusters
print()

kmeans = KMeans(n_clusters=cluster_num, random_state=42,)  
labels = kmeans.fit_predict(embeddings)
print() 

# print the Output 
for sentence, lable in zip(list_of_sentences, labels):
    print(f'sentence: {sentence} -> Cluster: {lable }')


# ===============================================================================================================
# ========================          VISUALIZATION of Embedding system       ===================================
tsne = TSNE(n_components=2, random_state=42, perplexity=n_sentences-2)  # n_components = 2 
reduced = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))  
plt.grid(True)

# colormap: ambil warna berbeda based on total of cluster
colormap = cm.get_cmap('tab20', cluster_num)

for i, (txt, label) in enumerate(zip(list_of_sentences, labels)):
    plt.scatter(reduced[i, 0], reduced[i, 1], color=colormap(label), s=60)
    plt.annotate(txt, (reduced[i, 0], reduced[i,1]), fontsize=8)

plt.title('Visualization of Multiple sentences & CLustering using SBERT', fontsize=12) 
plt.show()
