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
    修改时间：2026-07-26
        1. 新增：日期查询工具，支持LLM自主调用
        2. 重构：RAG链改为Agent架构，LLM自主决定调用知识库检索或日期工具(V_0.3)
        3. 修复：弃用 AgentExecutor + RunnableWithMessageHistory，改用 bind_tools 手动管理
    修改时间：2026-07-28
            1. 新增：邮件发送工具，支持LLM自主调用
'''

import os
import time
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain 相关导入
from langchain_chroma import Chroma
from langchain_zhipu import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage, ToolMessage, HumanMessage, SystemMessage
)


import models, functionTools
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

# ================= 3. 定义工具 =================

@tool
def get_current_date(query: str = "") -> str:
    """获取当前的日期和星期信息。当用户询问今天几号、今天是星期几、现在是什么时间等日期相关问题时，调用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return f"今天是{now.year}年{now.month}月{now.day}日，{weekday}。当前时间{now.hour}:{now.minute:02d}。"

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气情况。当用户询问天气、温度、是否下雨、穿什么衣服等问题时，调用此工具。参数city为城市名称。
    
    """
    weacher =  functionTools.get_weather(city)
    return weacher

@tool
def track_package(express_name, tracking_number: str) -> str:
    """
    查询快递物流信息。当用户询问快递到哪了、物流进度、包裹状态等问题时，调用此工具。
    参数 express_name为快递公司名称, tracking_number为快递单号。
    
    """
    package =  functionTools.get_package(express_name, tracking_number)
    return package

    
@tool  # 使用@tool装饰器标记此函数为一个工具
def search_knowledge_base(query: str) -> str:
    """在学校知识库中搜索相关信息。当用户询问学校规章制度、课程安排、招生信息、专业设置、校园生活等学校相关问题时，调用此工具。"""
    docs = retriever.invoke(query)
    if not docs:
        return "未在知识库中找到相关内容。"
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件。当用户要求发邮件、发送通知、邮件通知某人时，调用此工具。
    参数：
    - to: 收件人邮箱地址（必填）
    - subject: 邮件主题（必填）
    - body: 邮件正文内容（必填）
    """
    functionTools.send_email(to,subject,body)


# 工具列表
tools = [get_current_date, get_weather, track_package, search_knowledge_base, send_email]

# 工具映射（用于手动执行工具调用）
tool_map = {
    "get_current_date": get_current_date,
    "get_weather": get_weather,
    "track_package": track_package,
    "search_knowledge_base": search_knowledge_base,
    "send_email": send_email
}

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)


# ================= 4. 会话历史管理 =================

store = {}
SESSION_TIMEOUT = 1800
MAX_HISTORY_MESSAGES = 10


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史，附带过期清理和长度截断"""
    current_time = time.time()

    expired = [
        k for k, v in store.items()
        if current_time - v["last_access"] > SESSION_TIMEOUT
    ]
    for k in expired:
        del store[k]

    if session_id not in store:
        store[session_id] = {
            "history": InMemoryChatMessageHistory(),
            "last_access": current_time
        }

    store[session_id]["last_access"] = current_time
    history = store[session_id]["history"]

    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]

    return history


# ================= 5. 工具调用核心逻辑 =================

SYSTEM_PROMPT = """你是一个学校的智能问答助手。你可以使用以下工具来回答用户的问题：

1. get_current_date - 查询当前日期和时间
2. get_weather - 查询指定城市的天气情况
3. track_package - 查询快递物流信息
4. search_knowledge_base - 在学校知识库中搜索信息
5, send_email - 发送邮件

请根据用户的问题自主判断需要调用哪个工具：
- 如果用户问的是日期、时间相关的问题，调用 get_current_date
- 如果用户问的是天气情况，调用 get_weather
- 如果用户问的是快递物流信息，调用 track_package
- 如果用户问的是学校相关的问题，调用 search_knowledge_base
- 当用户要求“发邮件”时，你**必须**调用 `send_email` 工具。绝对不允许直接用文字回复“已发送”或产生幻觉！
- 调用 `send_email` 前，检查是否具备三个参数：收件人邮箱(to)、主题(subject)、正文(body)。
- 如果用户提供的信息不完整（例如只说了“发邮件给张三”，没给邮箱地址或正文），你**必须先追问用户**补充信息，不要盲目调用工具。
- 发送邮件前，向用户确认收件人、主题和正文内容，确保无误后再发送，调用 `send_email` 工具。
- 如果用户想把查询学校相关的问题的结果，通过邮件发送出去，请先确认发送的内容，等待用户确认后，调用 `send_email` 工具发送邮件。
- 如果三者都需要，可以依次调用多个工具

回答规则：
- 如果知识库中没有相关信息，请直接回答"抱歉，我的知识库中暂时没有关于这个问题的信息。"，不要编造答案
- 回答要简洁、准确、有条理"""

MAX_TOOL_ITERATIONS = 3


def chat_with_tools(question: str, history: InMemoryChatMessageHistory) -> str:
    """
    处理用户问题，支持工具调用和多轮对话。
    
    流程：
    1. 构建消息列表（系统提示 + 历史 + 当前问题）
    2. LLM 决定是否调用工具
    3. 如果调用工具，执行工具并将结果返回给 LLM
    4. LLM 根据工具结果生成最终回答
    5. 将对话保存到历史记录
    """
    # 构建消息列表
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=question))

    # 第一轮：LLM 决定是否调用工具
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    # 工具调用循环
    iterations = 0
    while ai_response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # 执行工具
            result = tool_map[tool_name].invoke(tool_args)
            print(f"  🔧 调用工具: {tool_name}({tool_args}) → {result}")

            # 将工具结果加入消息列表
            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            )
            messages.append(tool_msg)

        # LLM 根据工具结果继续生成
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)
        iterations += 1

    # 保存到历史记录
    history.add_message(HumanMessage(content=question))
    history.add_message(ai_response)

    return ai_response.content


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


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"


@app.post("/api/chat")
async def chat_with_bot(req: QueryRequest):
    """问答接口，支持多轮对话 + 工具调用"""
    history = get_session_history(req.session_id)

    try:
        answer = chat_with_tools(req.question, history)
        return {"answer": answer}
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return {"answer": f"抱歉，处理您的问题时出现了错误：{str(e)}"}


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
            * { box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0; display: flex; justify-content: center; height: 100vh; }
            .chat-container { width: 100%; max-width: 850px; background: #fff; display: flex; flex-direction: column; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
            /* ========== Header 居中 + 渐变背景 ========== */
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 20px;
                text-align: center;
                position: relative;
            }
            .header-title { font-size: 1.2em; font-weight: bold; }
            .header-info { font-size: 0.78em; opacity: 0.85; margin-top: 3px; }
            .clear-btn {
                position: absolute;
                right: 16px;
                top: 50%;
                transform: translateY(-50%);
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 6px 14px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.8em;
                transition: all 0.2s;
            }
            .clear-btn:hover { background: rgba(255,255,255,0.35); }
            /* ========== 消息区域 ========== */
            .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
            .message { padding: 10px 16px; border-radius: 12px; max-width: 80%; line-height: 1.6; word-wrap: break-word; }
            .user { align-self: flex-end; background-color: #667eea; color: white; border-bottom-right-radius: 4px; }
            .bot { align-self: flex-start; background-color: #f0f2f5; color: #333; border-bottom-left-radius: 4px; }
            /* ========== 智能体标签 ========== */
            .agent-tag { display: inline-block; font-size: 0.72em; padding: 3px 10px; border-radius: 12px; margin-bottom: 6px; font-weight: 600; background: #e3f2fd; color: #1565c0; }
            /* ========== 输入区 ========== */
            .input-area { display: flex; padding: 15px 20px; border-top: 1px solid #e0e0e0; background: #fff; gap: 10px; }
            input { flex: 1; padding: 10px 16px; border: 1px solid #ddd; border-radius: 24px; outline: none; font-size: 1em; transition: border-color 0.2s; }
            input:focus { border-color: #667eea; }
            button { padding: 10px 24px; background: #667eea; color: white; border: none; border-radius: 24px; cursor: pointer; font-size: 1em; transition: background 0.2s; }
            button:hover { background: #5568d3; }
            button:disabled { background: #b0b0b0; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="header">
                <div class="header-title">🎓 上海工业技术学校智能问答系统 (V_0.4)</div>
                <div class="header-info">🏫 学校客服 · 招生政策 · 课程安排 · 校规制度</div>
                <button class="clear-btn" onclick="clearChat()">🧹 新对话</button>
            </div>
            <div class="messages" id="messages">
                <div class="message bot">
                    <span class="agent-tag">🏫 学校客服</span><br>
                    你好！我是学校智能助手，你可以问我关于学校规章制度、课程安排等问题，也可以把查询结果发送邮件给你或者查天气和快递哦！
                </div>
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
                appendMessage('<em>正在查询中，请稍等...</em>', 'bot', loadingId, true);

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question, session_id: sessionId })
                    });
                    const data = await response.json();
                    updateMessage(loadingId, marked.parse(data.answer), true);
                } catch (error) {
                    updateMessage(loadingId, '❌ 请求失败，请检查后端服务是否正常运行。', true);
                } finally {
                    sendBtn.disabled = false;
                    userInput.focus();
                }
            }

            async function clearChat() {
                if (confirm("确定要清空当前对话记录，开启新对话吗？")) {
                    try {
                        await fetch('/api/clear_history', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: sessionId })
                        });
                    } catch (e) {
                        console.log('清除历史请求失败，忽略');
                    }
                    sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
                    localStorage.setItem('chat_session_id', sessionId);
                    messagesDiv.innerHTML = '<div class="message bot"><span class="agent-tag">🏫 学校客服</span><br>你好！我是学校智能助手，我们可以开始新的对话了。</div>';
                }
            }

            function appendMessage(text, sender, id = null, withTag = false) {
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${sender}`;
                if (id) msgDiv.id = id;
                if (withTag) {
                    msgDiv.innerHTML = '<span class="agent-tag">🏫 学校客服</span><br>' + text;
                } else {
                    msgDiv.innerHTML = text;
                }
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function updateMessage(id, text, withTag = false) {
                const msgDiv = document.getElementById(id);
                if (msgDiv) {
                    if (withTag) {
                        msgDiv.innerHTML = '<span class="agent-tag">🏫 学校客服</span><br>' + text;
                    } else {
                        msgDiv.innerHTML = text;
                    }
                }
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
