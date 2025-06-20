import os
from dotenv import load_dotenv
# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
load_dotenv()
from langchain_deepseek import ChatDeepSeek


# 检查API密钥是否已设置
def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
    return key

# 3.LLM 初始化
def deepseek_llm():
    api_key = get_deepseek_key()
    if not api_key:
        raise ValueError("没有ds key")
    return ChatDeepSeek(
        model = "deepseek-chat",
        # temperature=0 表示我们希望Agent的思考过程尽可能稳定和可复现
        temperature=0, 
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )
    