from flask import Flask, render_template_string, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import requests

load_dotenv()

app = Flask(__name__)

# Global variables
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None
retriever = None
groq_api_key = None

def initialize_groq():
    global groq_api_key
    groq_api_key = os.getenv("GROQ_API_KEY")
    return "Groq API key found" if groq_api_key else "Groq API key not found"

def initialize_vectorstore():
    global vectorstore, retriever
    
    try:
        vectorstore = FAISS.load_local("atomcamp_vector_db", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        return "Vectorstore loaded successfully"
    except:
        sample_data = [
            {
                "text": "Atomcamp is a leading data science education platform offering comprehensive courses in machine learning, Python programming, data analysis, and AI. We provide hands-on projects, expert mentorship, and career guidance to help students become successful data scientists.",
                "url": "https://www.atomcamp.com/about"
            },
            {
                "text": "Our courses include: Python for Data Science, Machine Learning Fundamentals, Deep Learning with TensorFlow, Data Visualization with Matplotlib and Seaborn, SQL for Data Analysis, Statistics for Data Science, and Advanced AI Techniques.",
                "url": "https://www.atomcamp.com/courses"
            },
            {
                "text": "Atomcamp offers flexible learning paths: Beginner Track (3 months) - Python basics, data manipulation, basic statistics. Intermediate Track (6 months) - Machine learning, advanced Python, real projects. Advanced Track (9 months) - Deep learning, AI, industry projects, job placement assistance.",
                "url": "https://www.atomcamp.com/learning-paths"
            }
        ]
        
        docs = [Document(page_content=item["text"], metadata={"url": item["url"]}) for item in sample_data]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("atomcamp_vector_db")
        retriever = vectorstore.as_retriever()
        
        return "Sample vectorstore created successfully"

def call_groq_api(message, context):
    global groq_api_key
    
    if not groq_api_key:
        return None
    
    try:
        system_prompt = f"""You are an AI assistant for Atomcamp, a data science education platform. 
        Use the following context to answer questions about Atomcamp's courses, career services, and data science topics.
        
        Context: {context}
        
        Guidelines:
        - Be helpful and informative
        - Focus on Atomcamp's offerings
        - Provide specific details when available
        - Use bullet points for lists
        - Keep responses concise but comprehensive
        - Do not use emojis
        """
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return None
            
    except Exception as e:
        return None

def generate_response(message, context):
    groq_response = call_groq_api(message, context)
    
    if groq_response:
        return groq_response
    
    # Fallback responses
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['course', 'courses', 'learn', 'study']):
        return """Atomcamp Courses:

Core Programs:
• Python for Data Science - Master Python programming fundamentals
• Machine Learning Fundamentals - Learn ML algorithms and applications  
• Deep Learning with TensorFlow - Build neural networks and AI models
• Data Visualization - Create stunning charts with Matplotlib & Seaborn
• SQL for Data Analysis - Database querying and data manipulation
• Statistics for Data Science - Statistical analysis and hypothesis testing

Learning Tracks:
• Beginner Track (3 months) - Perfect for newcomers
• Intermediate Track (6 months) - Build real-world projects
• Advanced Track (9 months) - Industry-ready with job placement

Would you like details about any specific course?"""
    
    elif any(word in message_lower for word in ['career', 'job', 'placement']):
        return """Career Services at Atomcamp:

Job Placement Support:
• Resume building and optimization
• Technical interview preparation
• Portfolio development guidance
• Direct connections with hiring partners
• Mock interviews with industry experts

Career Growth:
• Average salary increase: 150-300%
• 95% job placement rate within 6 months
• Access to exclusive job opportunities
• Ongoing career mentorship
• Industry networking events

Ready to transform your career in data science?"""
    
    else:
        return f"""Thank you for your question about "{message}"!

As an Atomcamp AI assistant, I'm here to help you with:

Course Information - Learn about our data science programs
Career Guidance - Job placement and career growth
Technical Topics - Python, ML, AI, and data analysis
Getting Started - How to begin your data science journey

Would you like me to elaborate on any specific aspect?"""

@app.route('/')
def index():
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>atomcamp AI Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f9fafb 0%, #ffffff 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding-top: 2rem;
            padding-bottom: 1.5rem;
        }

        .logo {
            margin-bottom: 1.5rem;
        }

        .logo-img {
            height: 48px;
            width: auto;
            object-fit: contain;
            max-width: 200px;
        }

        .title-bubble {
            width: 100%;
            max-width: 64rem;
            padding: 0 1rem;
        }

        .bubble-container {
            background: #f0fdf4;
            border: 1px solid #e5e7eb;
            border-radius: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            max-width: fit-content;
        }

        .bubble-content {
            text-align: left;
        }

        .chatbot-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #166534;
            margin-bottom: 0.25rem;
            letter-spacing: -0.025em;
        }

        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .typing-dots {
            display: flex;
            gap: 0.25rem;
        }

        .typing-dot {
            width: 0.375rem;
            height: 0.375rem;
            background: #166534;
            border-radius: 50%;
            animation: bounce 1s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.1s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.2s;
        }

        .typing-text {
            color: #166534;
            font-size: 1rem;
            font-weight: 500;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: #16a34a;
        }

        .status-text {
            font-size: 0.875rem;
            font-weight: 500;
            color: #166534;
        }

        .chat-container {
            max-width: 64rem;
            margin: 0 auto;
            padding: 0 1rem;
            padding-bottom: 8rem;
            flex: 1;
            overflow-y: auto;
            height: calc(100vh - 300px);
        }

        .welcome-screen {
            text-align: center;
            padding: 3rem 0;
        }

        .welcome-card {
            background: white;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 2rem;
            max-width: 36rem;
            margin: 0 auto;
        }

        .welcome-icon {
            width: 3rem;
            height: 3rem;
            background: #22c55e;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }

        .welcome-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.75rem;
            letter-spacing: -0.025em;
        }

        .welcome-description {
            color: #6b7280;
            font-size: 0.875rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 0.75rem;
            text-align: left;
        }

        .feature-card {
            background: #f9fafb;
            border-radius: 0.5rem;
            padding: 0.75rem;
        }

        .feature-title {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }

        .feature-list {
            font-size: 0.75rem;
            color: #6b7280;
            line-height: 1.5;
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .feature-list li {
            margin-bottom: 0.25rem;
        }

        .messages-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .message {
            display: flex;
            animation: fadeIn 0.3s ease-out;
            margin-bottom: 0.5rem;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.assistant {
            justify-content: flex-start;
        }

        .message-content {
            display: flex;
            align-items: flex-end;
            gap: 0.5rem;
            max-width: 90%;
        }

        .avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .avatar.user {
            background: #22c55e;
            color: white;
        }

        .avatar.assistant {
            background: #e5e7eb;
            color: #6b7280;
        }

        .message-bubble {
            padding: 0.625rem 0.875rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            line-height: 1.4;
            max-width: 90%;
        }


        .message.user .message-bubble {
            background: #22c55e;
            color: white;
            border-bottom-right-radius: 0.375rem;
        }

        .message.assistant .message-bubble {
            background: #e5e7eb;
            color: #1f2937;
            border-bottom-left-radius: 0.375rem;
        }

        .message-text {
            font-size: 0.875rem;
            font-weight: 500;
            line-height: 1.5;
        }

        .message-formatted {
            font-size: 0.875rem;
            line-height: 1.5;
        }

        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(8px);
            border-top: 1px solid #e5e7eb;
            padding: 1rem;
        }

        .input-container {
            max-width: 64rem;
            margin: 0 auto;
        }

        .input-form {
            position: relative;
        }

        .input-wrapper {
            position: relative;
            background: #4b5563;
            border-radius: 9999px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .message-input {
            width: 100%;
            padding: 0.875rem 1.25rem;
            padding-right: 3rem;
            background: transparent;
            border: none;
            outline: none;
            color: white;
            font-size: 0.875rem;
            font-weight: 500;
            resize: none;
            line-height: 1.5;
            min-height: 3.25rem;
            max-height: 9rem;
            overflow-y: auto;
        
        }


        .message-input::placeholder {
            color: #d1d5db;
        }

        .send-button {
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            background: #22c55e;
            color: white;
            border: none;
            border-radius: 50%;
            width: 2.5rem;
            height: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }

        .send-button:hover:not(:disabled) {
            background: #16a34a;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
            padding: 0 0.5rem;
        }

        .connection-status {
            font-size: 0.75rem;
            font-weight: 500;
            color: #6b7280;
        }

        .char-count {
            font-size: 0.75rem;
            font-weight: 500;
            color: #9ca3af;
        }

        .hidden {
            display: none !important;
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-4px);
            }
        }


        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 768px) {
            .header {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            
            .chat-container {
                padding-bottom: 6rem;
            }
            
            .message-content {
                max-width: calc(100vw - 2rem);
            }
            
            .message.user .message-content {
                max-width: calc(100vw - 2rem);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <img src="/static/atomcamp_logo.png" alt="Atomcamp Logo" class="logo-image" style="height: 42px;">
        </div>
        
        <div class="title-bubble">
            <div class="bubble-container">
                <div class="bubble-content">
                    <h1 class="chatbot-title">Chatbot</h1>
                    
                    <div id="typingIndicator" class="typing-indicator hidden">
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                        <span class="typing-text">Typing...</span>
                    </div>
                    
                    <div id="statusIndicator" class="status-indicator">
                        <div class="status-dot"></div>
                        <span class="status-text">Online</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="chat-container">
        <div id="welcomeScreen" class="welcome-screen">
            <div class="welcome-card">
                <div class="welcome-icon">✨</div>
                <h2 class="welcome-title">Welcome to atomcamp AI</h2>
                <p class="welcome-description">Ask me about courses, data science concepts, and learning paths.</p>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3 class="feature-title">Ask me about:</h3>
                        <ul class="feature-list">
                            <li>• Course information</li>
                            <li>• Data science concepts</li>
                            <li>• Learning paths</li>
                        </ul>
                    </div>
                    <div class="feature-card">
                        <h3 class="feature-title">Try asking:</h3>
                        <ul class="feature-list">
                            <li>• "What courses do you offer?"</li>
                            <li>• "Explain machine learning"</li>
                            <li>• "How do I get started?"</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="messagesContainer" class="messages-container hidden"></div>
    </div>

    <div class="input-area">
        <div class="input-container">
            <form id="chatForm" class="input-form">
                <div class="input-wrapper">
                    <textarea 
                    id="messageInput" 
                    class="message-input"
                    placeholder="Ask me anything about atomcamp..."
                    maxlength="1000"
                    autocomplete="off"
                    rows="1"
                    >
                    </textarea>

                    <button type="submit" id="sendButton" class="send-button">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 19V5M5 12l7-7 7 7"/>
                        </svg>
                    </button>
                </div>
                
                <div class="status-bar">
                    <span id="connectionStatus" class="connection-status">Connected to atomcamp AI</span>
                    <span id="charCount" class="char-count">0/1000</span>
                </div>
            </form>
        </div>
    </div>

    <script>
        let messages = [];
        let isTyping = false;
        let isConnected = true;

        const welcomeScreen = document.getElementById('welcomeScreen');
        const messagesContainer = document.getElementById('messagesContainer');
        const chatForm = document.getElementById('chatForm');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const statusIndicator = document.getElementById('statusIndicator');
        const connectionStatus = document.getElementById('connectionStatus');
        const charCount = document.getElementById('charCount');

        chatForm.addEventListener('submit', handleSubmit);
        messageInput.addEventListener('input', updateCharCount);

        messageInput.addEventListener('input', () => {
           messageInput.style.height = 'auto';
           messageInput.style.height = messageInput.scrollHeight + 'px';
        });

        function updateCharCount() {
            const count = messageInput.value.length;
            charCount.textContent = `${count}/1000`;
        }

        async function handleSubmit(e) {
            e.preventDefault();
            const message = messageInput.value.trim();
            
            if (!message || isTyping) return;
            
            addMessage(message, 'user');
            messageInput.value = '';
            updateCharCount();
            setTyping(true);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                await new Promise(resolve => setTimeout(resolve, 1200));
                
                addMessage(data.response || 'Sorry, I could not generate a response.', 'assistant');
                setConnected(true);
                
            } catch (error) {
                console.error('Error:', error);
                setConnected(false);
                addMessage('I am having trouble connecting. Please try again.', 'assistant');
            } finally {
                setTyping(false);
            }
        }

        function addMessage(content, role) {
            if (messages.length === 0) {
                welcomeScreen.classList.add('hidden');
                messagesContainer.classList.remove('hidden');
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            const avatar = document.createElement('div');
            avatar.className = `avatar ${role}`;
            
            if (role === 'user') {
                avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>';
            } else {
                avatar.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>';
            }
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            if (role === 'user') {
                bubble.innerHTML = `<div class="message-text">${content}</div>`;
                messageContent.appendChild(bubble);
                messageContent.appendChild(avatar);
            } else {
                bubble.innerHTML = `<div class="message-formatted">${formatMessageContent(content)}</div>`;
                messageContent.appendChild(avatar);
                messageContent.appendChild(bubble);
            }
            
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            scrollToBottom();
            messages.push({ content, role, timestamp: new Date() });
        }

        function formatMessageContent(content) {
            return content.split('\\n').map(line => {
                if (line.trim().startsWith('•') || line.trim().startsWith('-')) {
                    return `<div style="display: flex; align-items: flex-start; margin-bottom: 0.375rem;">
                        <span style="color: #16a34a; margin-right: 0.5rem; margin-top: 0.125rem; font-size: 0.875rem; font-weight: 500;">•</span>
                        <span style="font-size: 0.875rem; line-height: 1.5;">${line.replace(/^[•-]\\s*/, '')}</span>
                    </div>`;
                } else if (/^\\d+\\./.test(line.trim())) {
                    const match = line.match(/^\\d+\\./);
                    return `<div style="display: flex; align-items: flex-start; margin-bottom: 0.375rem;">
                        <span style="color: #16a34a; margin-right: 0.5rem; font-weight: 600; font-size: 0.875rem;">${match ? match[0] : ''}</span>
                        <span style="font-size: 0.875rem; line-height: 1.5;">${line.replace(/^\\d+\\.\\s*/, '')}</span>
                    </div>`;
                } else if (line.trim() === '') {
                    return '<br>';
                } else {
                    return `<p style="margin-bottom: 0.375rem; font-size: 0.875rem; line-height: 1.5;">${line}</p>`;
                }
            }).join('');
        }

        function setTyping(typing) {
            isTyping = typing;
            sendButton.disabled = typing;
            
            if (typing) {
                typingIndicator.classList.remove('hidden');
                statusIndicator.classList.add('hidden');
            } else {
                typingIndicator.classList.add('hidden');
                statusIndicator.classList.remove('hidden');
            }
        }

        function setConnected(connected) {
            isConnected = connected;
            connectionStatus.textContent = connected ? 'Connected to atomcamp AI' : 'Connection lost';
        }

        function scrollToBottom() {
            setTimeout(() => {
                window.scrollTo({
                    top: document.body.scrollHeight,
                    behavior: 'smooth'
                });
            }, 100);
        }
    </script>
</body>
</html>
    """
    return render_template_string(html_template)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if retriever:
            docs = retriever.get_relevant_documents(message)
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
        else:
            context = "I'm an AI assistant for Atomcamp, a data science education platform."
        
        response = generate_response(message, context)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Initialize systems
initialize_groq()
initialize_vectorstore()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
