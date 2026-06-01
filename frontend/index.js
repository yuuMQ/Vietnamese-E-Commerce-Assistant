const sessionId = "user_html_session_" + Math.floor(Math.random() * 100000);
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function handleKeyPress(event) {
    if (event.key == 'Enter') {
        sendMessage();
    }
}
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    userInput.value = '';

    const loadingId = appendMessage('Dạ, mình đợi shop xíu ạ...', 'bot loading');
    scrollToBottom();

    try {
        const response = await fetch('http://127.0.0.1:8000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                session_id: sessionId
            })
        });

        document.getElementById(loadingId).remove();

        if (response.ok) {
            const data = await response.json();
            appendMessage(data.response, 'bot');
        } else {
            appendMessage('Dạ, kết nối hệ thống có chút trục trặc, bạn thử lại giúp mình ạ.', 'bot');
        }
    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage('Dạ, không kết nối được với Server Backend. Bạn đã chạy file app.py chưa ạ?', 'bot');
    }

    scrollToBottom();
}

function appendMessage(text, senderClass) {
    const messageDiv = document.createElement('div');
    const id = 'msg_' + Math.random().toString(36).substr(2, 9);
    messageDiv.id = id;
    messageDiv.className = `message ${senderClass}`;
    messageDiv.innerText = text;
    chatMessages.appendChild(messageDiv);
    return id;
}