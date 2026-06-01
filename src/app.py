import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import ECommerceAssistant
from fastapi.middleware.cors import CORSMiddleware
import webbrowser
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title='E-Commerce Chatbot RAG')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


chatbot = ECommerceAssistant() # Init Assistant
sessions_memory = {} # Temp history

class ChatRequest(BaseModel):
    message: str
    session_id: str = 'default_user'


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")
@app.get("/")
def read_root():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"status": "error", "message": "Không tìm thấy file index.html"}

@app.post('/api/chat')
async def chat_endpoint(request: ChatRequest):
    user_query = request.message.strip()
    session_id = request.session_id.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Tin nhắn trống.")

    try:
        if session_id not in sessions_memory:
            sessions_memory[session_id] = []
        current_history = sessions_memory[session_id]

        reply_text = chatbot.process_query(user_query, current_history)

        current_history.append(HumanMessage(content=user_query))
        current_history.append(AIMessage(content=reply_text))

        return {"response": reply_text, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
