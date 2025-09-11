# ===========================              MINI PROJECT OF EMBEDDING SYSTEM          ============================
#                                            Semantic FAQ Search

'''
sebenarnya ini lebih mirip kearah tanya-jawab chatbot!

How to use ? 
kek contoh nya kita tanya something, dan system bakal cari tau dari pertanyaan yg diajukan oleh user, 
apakah faq_data ada yg sama dengan apa yg telah "dirumuskan" didalam system itu sendiri, sehingga secara otomatis, 
system mencari jawaban yg atleast agak mirip (another term: SIMILARITY) dengan apa yg telah dirumuskan, 
nah apabila ada, maka system langsung menjawab jawaban yg SIMILAR dengan pertanyaan yg telah diajukan oleh user
'''

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import numpy as np

model =  SentenceTransformer('all-MiniLM-L6-v2')

faq_data = [
    {
        "question": "Bagaimana cara reset password?", 
        "answer": "Klik 'Lupa Password' lalu ikuti petunjuknya. Kamu juga bisa pergi ke pengaturan, dan tekan \"Lupa Password\"."},
    {
        "question": "Apa itu langganan premium?",
        "answer": "Langganan premium memberi akses ke fitur eksklusif. So, kamu bisa mendapatkan akses lebih selama 1 bulan "},
    {
        "question": "Bagaimana cara mengganti email?", 
        "answer": "Masuk ke pengaturan akun lalu ubah email, lalu tekan \"Ubah alamat email\". "},
    {
        "question": "Apakah bisa membatalkan pesanan?", 
        "answer": "Bisa, jika pesanan belum dikirim. Namun, sebelum membatalkan pesanan, mohon teliti terlebih dahulu terhadap apa yang telah anda pesan"}, 
    {
        'question': 'Apakah akan mendapatkan bonus, jika mengundang atau mengajak teman untuk menggunakan aplikasi ini?', 
        'answer': 'Pastinya! kamu akan mendapatkan bonus voucher belanja sebesar Rp. XXX'
    },
    {
        'question':'apakah akan ada event atau acara atau agenda dalam waktu terdekat yang diadakan oleh company atau aplikasi ini?', 
        'answer': 'Untuk acara dan event dalam waktu terdekat, tetap staytune di aplikasi AAAA ini ya!! '
    }
]

# embedding all question in FAQ
faq_questions = [item['question'] for item in faq_data]
faq_embeddings = model.encode(faq_questions)

# INPUT from USER
def find_best_answer(user_question):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0] 
    best_idx = np.argmax(similarities) 
    
    return faq_data[best_idx]['answer']

# example 
stop_list = ['stop', 'exit', 'quit']
while True:
    # user_quest = 'gimana cara ganti email akun saya?'
    user_quest = input('\nWanna ask something? Just feel free to ask me :) \n')
    if user_quest in stop_list: 
        break 
    else: 
        answer = find_best_answer(user_quest) 
        print('\nAnswer:\n', answer)
        print('\n======================================================\n') 

# ===========================================================================
#                               VISUALIZATION PART 

# sentences = faq_questions + [user_quest] 
# embeddings = model.encode(sentences)  

# tsne = TSNE(n_components=2, perplexity=4, random_state=42)  
# points = tsne.fit_transform(embeddings) 

# plt.figure(figsize=(8, 5)) 
# plt.grid()
# for i , label in enumerate(sentences): 
#     plt.scatter(points[i][0], points[i][1]) 
#     plt.annotate(label, (points[i][0] +.1, points[i][1] +.1)) 
# plt.title('Visualisasi FAQ Embedding + User question') 
# plt.show()