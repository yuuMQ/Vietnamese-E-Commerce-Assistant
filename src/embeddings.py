# Embedding and Vector Database
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch

embedding_function = HuggingFaceEmbeddings(model_name='keepitreal/vietnamese-sbert')

dataset = load_from_disk('../dataset')

langchain_documents = []
for index, item in enumerate(dataset['train']):
    conversation_text = ''
    for turn in item['conversations']:
        role = 'Khách hàng' if turn['from'] == 'human' else 'Nhân viên'
        conversation_text += f'{role}: {turn['value']}\n'

    # print(conversation_text)
    doc = Document(page_content=conversation_text.strip(), metadata={'row_id': index})
    langchain_documents.append(doc)

vector_db = Chroma.from_documents(
    documents=langchain_documents,
    embedding=embedding_function,
    persist_directory='../chroma_vectorstore'
)
print('Vector store created successfully')
