from sentence_transformers import SentenceTransformer as st
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.cluster import KMeans


# Model: text to embedding 
model = st('all-MiniLM-L6-v2') 

sentences = [
    'Embedding is an amazing method!',
    'AI needs to understand semantic relationships',
    'one of the ai method is embedding system',
    'The weather is sunny and warm today',
    'It is a bright and hot day',
    'Freezing conditions with ice formation expected',
    'Climate change affects global weather patterns',
    'Global warming is altering traditional seasons',
    'Weather scientists analyze climate data'
]

total_sentences = len(sentences) 
cluster_num =2
# print('total n sentences:', total_sentences)

embeddings = model.encode(sentences) 
result = embeddings.shape

print()
print('this is the num of embedding shape:', result)

res_of_embed1 = embeddings[0]
res_of_embed2 = embeddings[1]

print('sentence:', sentences[0], '\nfirst embedded sentence:\n', res_of_embed1[:10])
print('\nsentence:', sentences[1], ' \nsecond embedded sentence:\n', res_of_embed2[:10])

# # ===================================================================
# #               VISUALIZATION  OF  EMBEDDING

tsne = TSNE(n_components=2, perplexity=total_sentences-2) 
embeddings_2d = tsne.fit_transform(embeddings)

# plotting 
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])  
plt.grid()
plt.show()

# kmeans = KMeans(n_clusters=cluster_num, random_state=42,)  
# labels = kmeans.fit_predict(embeddings)

# tsne = TSNE(n_components=2, random_state=42, perplexity=total_sentences-2)  # n_components = 2 
# reduced = tsne.fit_transform(embeddings)

# plt.figure(figsize=(10, 6))  
# plt.grid(True)

# # colormap: ambil warna berbeda based on total of cluster
# colormap = cm.get_cmap('tab20', cluster_num)

# for i, (txt, label) in enumerate(zip(sentences, labels)):
#     plt.scatter(reduced[i, 0], reduced[i, 1], color=colormap(label), s=60)
#     plt.annotate(txt, (reduced[i, 0], reduced[i,1]), fontsize=8)

# plt.title('Visualization of Multiple sentences & CLustering using SBERT', fontsize=12) 
# plt.show()
