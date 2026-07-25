'''
    创建时间：2026-07-20
    功能：
        1. 创建一个基于 LangChain 的 RAG 系统，用于学校智能问答
        2. 使用 Chroma 向量库和 ZhipuAi 模型实现问答
        3. 增加多轮对话短期记忆功能 (V_0.2)
    修改时间：2026-07-22
        1. 修复：检索不感知历史上下文 —— 增加问题重写链
        2. 修复：store 内存泄漏 —— 增加过期清理 + 清除历史接口
        3. 修复：历史消息无截断 —— 限制保留最近10条消息
'''

import os
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain 相关导入
from langchain_chroma import Chroma
from langchain_zhipu import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import models
from uploadRouter import router as upload_router
from manageRouter import router as manage_router

# ================= 1. 配置参数 =================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
PERSIST_DIRECTORY = "./chroma_db_school"

# ================= 2. 初始化 RAG 组件 =================
print("正在加载向量数据库和模型，请稍候...")

embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=ZHIPU_API_KEY)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name="school_knowledge"
)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = models.llm_Zhipu()

# ================= 3. 会话历史管理（含过期清理 + 长度截断）=================

# store 结构: {session_id: {"history": InMemoryChatMessageHistory(), "last_access": timestamp}}
store = {}

# 过期时间（秒），30分钟未访问自动清理
SESSION_TIMEOUT = 1800

# 最大保留消息条数（5轮对话 = 10条消息）
MAX_HISTORY_MESSAGES = 10


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史，附带过期清理和长度截断"""
    current_time = time.time()

    # 清理过期的会话
    expired = [
        k for k, v in store.items()
        if current_time - v["last_access"] > SESSION_TIMEOUT
    ]
    for k in expired:
        del store[k]

    # 创建或获取会话
    if session_id not in store:
        store[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "last_access": current_time
        }

    store[session_id]["last_access"] = current_time
    history = store[session_id]["history"]

    # 截断历史，只保留最近 MAX_HISTORY_MESSAGES 条
    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]

    return history


# ================= 4. 问题重写链（历史感知）=================

condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个问题改写助手。根据以下【对话历史】，将用户的【最新问题】改写为一个独立的、完整的问题，"
     "使其不依赖上下文也能被理解。直接输出改写后的问题，不要加任何解释或前缀。\n\n"
     "示例：\n"
     "历史：用户问'学校的招生电话是多少？'，助手回答'021-12345678'。\n"
     "最新问题：那地址呢？\n"
     "改写结果：学校的地址是什么？"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

condense_question_chain = (
    condense_question_prompt
    | llm
    | StrOutputParser()
)

# ================= 5. 构建 RAG 链 =================

# 辅助函数：将检索到的文档列表格式化为字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 主 Prompt（包含历史占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个学校的智能问答助手。请根据以下提供的【已知上下文】来回答用户的【问题】。"
     "如果上下文中没有包含足够的信息来回答问题，请直接回答"
     "\"抱歉，我的知识库中暂时没有关于这个问题的信息。\"，不要编造答案。\n\n"
     "【已知上下文】：\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# 构建基础 RAG 链（检索时使用改写后的独立问题）
rag_chain = (
    RunnablePassthrough.assign(
        # 第一步：结合历史改写问题
        standalone_question=lambda x: condense_question_chain.invoke(
            {"question": x["question"], "chat_history": x.get("chat_history", [])}
        )
    )
    .assign(
        # 第二步：用改写后的问题去检索
        context=(lambda x: x["standalone_question"]) | retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 使用 RunnableWithMessageHistory 包装，赋予记忆能力
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

print("✅ 系统初始化完成！")

# ================= 6. FastAPI 后端逻辑 =================
app = FastAPI(title="上海工业技术学校智能问答系统")

app.include_router(upload_router)
app.include_router(manage_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"


@app.post("/api/chat")
async def chat_with_bot(req: QueryRequest):
    """问答接口，支持多轮对话"""
    config = {"configurable": {"session_id": req.session_id}}

    answer = conversational_rag_chain.invoke(
        {"question": req.question},
        config=config,
    )
    return {"answer": answer}


@app.post("/api/clear_history")
async def clear_history(req: QueryRequest):
    """清除指定会话的历史记录"""
    if req.session_id in store:
        del store[req.session_id]
        return {"status": "ok", "message": "历史记录已清除"}
    return {"status": "ok", "message": "无历史记录可清除"}


# ================= 7. 前端 Web 页面 =================
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>上海工业技术学校智能问答系统</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f5f5; margin: 0; display: flex; justify-content: center; height: 100vh; }
            .chat-container { width: 100%; max-width: 800px; background: #fff; display: flex; flex-direction: column; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .header { background: #007bff; color: white; padding: 15px; text-align: center; font-size: 1.2em; font-weight: bold; position: relative; }
            .clear-btn { position: absolute; right: 15px; top: 50%; transform: translateY(-50%); background: #dc3545; color: white; border: none; padding: 6px 12px; border-radius: 5px; cursor: pointer; font-size: 0.8em; }
            .clear-btn:hover { background: #c82333; }
            .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { padding: 10px 15px; border-radius: 10px; max-width: 80%; line-height: 1.5; word-wrap: break-word; }
            .user { align-self: flex-end; background-color: #007bff; color: white; }
            .bot { align-self: flex-start; background-color: #e9ecef; color: #333; }
            .input-area { display: flex; padding: 15px; border-top: 1px solid #ddd; background: #fff; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; outline: none; font-size: 1em; }
            button { margin-left: 10px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 20px; cursor: pointer; font-size: 1em; }
            button:disabled { background: #aaa; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="header">
                🎓 上海工业技术学校智能问答系统 (V_0.2)
                <button class="clear-btn" onclick="clearChat()">🧹 新对话</button>
            </div>
            <div class="messages" id="messages">
                <div class="message bot">你好！我是学校智能助手，你可以问我关于学校规章制度、课程安排等问题。我具备短期记忆，可以进行多轮对话哦！</div>
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="请输入你的问题..." onkeypress="handleKeyPress(event)">
                <button id="sendBtn" onclick="sendMessage()">发送</button>
            </div>
        </div>

        <script>
            const messagesDiv = document.getElementById('messages');
            const userInput = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');

            let sessionId = localStorage.getItem('chat_session_id');
            if (!sessionId) {
                sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('chat_session_id', sessionId);
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') sendMessage();
            }

            async function sendMessage() {
                const question = userInput.value.trim();
                if (!question) return;

                appendMessage(question, 'user');
                userInput.value = '';
                sendBtn.disabled = true;

                const loadingId = 'loading-' + Date.now();
                appendMessage('<em>正在检索知识库并思考中...</em>', 'bot', loadingId);

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question, session_id: sessionId })
                    });
                    const data = await response.json();
                    updateMessage(loadingId, marked.parse(data.answer));
                } catch (error) {
                    updateMessage(loadingId, '❌ 请求失败，请检查后端服务是否正常运行。');
                } finally {
                    sendBtn.disabled = false;
                    userInput.focus();
                }
            }

            async function clearChat() {
                if (confirm("确定要清空当前对话记录，开启新对话吗？")) {
                    // 通知后端清除历史
                    try {
                        await fetch('/api/clear_history', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: sessionId })
                        });
                    } catch (e) {
                        console.log('清除历史请求失败，忽略');
                    }
                    // 生成新的 session_id
                    sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
                    localStorage.setItem('chat_session_id', sessionId);
                    // 清空界面
                    messagesDiv.innerHTML = '<div class="message bot">你好！我是学校智能助手，我们可以开始新的对话了。</div>';
                }
            }

            function appendMessage(text, sender, id = null) {
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${sender}`;
                if (id) msgDiv.id = id;
                msgDiv.innerHTML = text;
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function updateMessage(id, text) {
                const msgDiv = document.getElementById(id);
                if (msgDiv) msgDiv.innerHTML = text;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ================= 8. 启动入口 =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
