# 
# !!!!   BERPOTENSI UNTUK DIJADIKAN mini project    !!!
# !!!!   BERPOTENSI UNTUK DIJADIKAN mini project    !!!
# !!!!   BERPOTENSI UNTUK DIJADIKAN mini project    !!!

import chromadb 
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.config import Settings

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name='my_collection')

collection.add(
    documents = [
        'i like african foods',
        'i like american foods',
        'i like asian foods', 
        'i like european foods',
        'i like indonesian foods', 
    ], 
    ids= ['id1', 'id2', 'id3', 'id4', 'id5',], 
)

# Query the collection Part
# =======================================================================

results = collection.query(
    query_texts=[
        'i love pizza, sushi, McDonalds'], 
    n_results=3, 
)

documents = results['documents']
distances = results['distances']

print()
print(results)
print('Documents:', documents)
print('Distances:', distances)
