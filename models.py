import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI

# 加载根目录下的 .env 文件
load_dotenv()

# 读取环境变量
api_key = os.getenv("ZHIPU_API_KEY")
model_name = os.getenv("ZHIPU_MODEL")

def llm_Zhipu():
    # 初始化GLM
    llm = ChatZhipuAI(
        zhipuai_api_key = api_key,
        model = model_name,
        temperature = 0.1,  # 0~1，越小输出越稳定
        max_tokens = 2048
    )
    return llm

