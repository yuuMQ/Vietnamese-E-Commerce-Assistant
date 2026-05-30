import os
import torch
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

API_KEY = 'YOUR_API_KEY'

if __name__ == '__main__':
    embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

    vector_db = Chroma(
        persist_directory='../chroma_vectorstore',
        embedding_function=embedding_function,
    )
    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0,
        api_key=API_KEY
    )


    chat_history_list = []

    def process_query(user_query, history_list):
        matched_docs = vector_db.similarity_search(user_query, k=4)

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

        response = llm.invoke(messages)

        return response.content.strip()

    print('RAG: ')
    print('--------------------------------------------------')
    while True:
        user_input = input('User: ')
        if user_input.lower() in ['exit', 'quit']:
            print('Goodbye!')
            break
        if not user_input.strip():
            continue

        reply = process_query(user_input, chat_history_list)
        print(f'Chatbot: {reply}')
        print('-' * 50)

        chat_history_list.append(HumanMessage(content=user_input))
        chat_history_list.append(AIMessage(content=reply))