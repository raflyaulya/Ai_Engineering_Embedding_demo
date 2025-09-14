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
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' and follow the instructions. You can also go to settings and tap \"Forgot Password\"."
    },
    {
        "question": "What is a premium subscription?",
        "answer": "A premium subscription gives you access to exclusive features. So, you can enjoy more access for 1 month."
    },
    {
        "question": "How do I change my email?",
        "answer": "Go to your account settings, update your email, and tap \"Change email address\"."
    },
    {
        "question": "Can I cancel an order?",
        "answer": "Yes, as long as the order hasn't been shipped yet. However, please carefully review your order details before cancelling."
    },  
    {
        "question": "Will I get a bonus for inviting friends to use this app?",
        "answer": "Absolutely! You'll receive a shopping voucher bonus worth Rp. XXX."
    },
    {
        "question": "Are there any upcoming events or agendas hosted by the company or this app?",
        "answer": "Stay tuned on the AAAA app for updates on upcoming events and agendas!"
    },
    {
        "question": "How do I upgrade or downgrade my subscription plan?",
        "answer": "Go to your subscription settings, and choose 'Change Plan' to upgrade or downgrade as needed."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact our support team via the 'Help Center' in the app or email us at support@example.com."
    },
    {
        "question": "Where can I view my order history?",
        "answer": "Go to 'My Orders' section in your profile to view all your past orders and their statuses."
    },
    {
        "question": "Can I change my delivery address after placing an order?",
        "answer": "If your order hasn’t shipped yet, yes! You can edit your delivery address in the 'My Orders' section."
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