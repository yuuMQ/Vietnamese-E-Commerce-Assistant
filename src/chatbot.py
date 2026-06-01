import os
import torch
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

API_KEY = 'YOUR_API_KEY'

class ECommerceAssistant:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name='keepitreal/vietnamese-sbert')
        self.vector_db = Chroma(
            persist_directory='../chroma_vectorstore',
            embedding_function=self.embedding_function,
        )
        self.llm = ChatGroq(
            model='llama-3.1-8b-instant',
            temperature=0,
            api_key=API_KEY
        )

    def process_query(self, user_query, history_list):
        matched_docs = self.vector_db.similarity_search(user_query, k=4)

        context_str = ''
        for doc in matched_docs:
            context_str += doc.page_content + '\n\n'

        messages = []

        system_content = f'''
                   Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp, luôn dùng kính ngữ 'dạ', 'ạ' và xưng hô 'mình' - 'bạn' lịch sự với khách hàng.
                   Hãy sử dụng THÔNG TIN THAM KHẢO được cung cấp dưới đây để trả lời câu hỏi của khách hàng một cách chính xác nhất. 
                   Nếu thông tin không có, hãy khéo léo từ chối và hẹn kiểm tra lại, tuyệt đối không tự bịa thông tin sản phẩm.

                   THÔNG TIN THAM KHẢO:
                   {context_str.strip()}
               '''

        messages.append(SystemMessage(content=system_content))
        messages.extend(history_list)
        messages.append(HumanMessage(content=user_query))

        response = self.llm.invoke(messages)

        return response.content.strip()

