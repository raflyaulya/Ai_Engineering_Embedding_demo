from sentence_transformers import SentenceTransformer as st
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# Model: text to embedding 
model = st('all-MiniLM-L6-v2') 

sentences = [
    'Embedding is an amazing method!',
    'AI needs to understand semantic relationships',
    'one of the ai method is embedding system',
    'ai engineering is the most famous topic today '
    # 'The weather is sunny and warm today',
    # 'It is a bright and hot day', 
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

# Define number of clusters
cluster_num = 2
kmeans = KMeans(n_clusters=cluster_num, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=total_sentences-2, random_state=42)
reduced = tsne.fit_transform(embeddings)
# Create plot
plt.figure(figsize=(12, 7))
plt.grid(True)
colormap = cm.get_cmap('tab10', cluster_num)

# Plot Convex Hull (arsiran)
for cluster_id in range(cluster_num):
    cluster_points = reduced[labels == cluster_id]
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        hull_pts = cluster_points[hull.vertices]
        plt.fill(hull_pts[:, 0], hull_pts[:, 1],
                 color=colormap(cluster_id),
                 alpha=0.2,
                 label=f'Cluster {cluster_id}')

# Scatter points + text annotation
for i, (txt, label) in enumerate(zip(sentences, labels)):
    plt.scatter(reduced[i, 0], reduced[i, 1], color=colormap(label), s=60)
    plt.annotate(txt, (reduced[i, 0]+0.5, reduced[i, 1]+0.5), fontsize=9)

plt.title('Visualization of Sentence Embeddings with Clustering using SBERT + t-SNE', fontsize=13)
plt.legend()
plt.tight_layout()
plt.show()