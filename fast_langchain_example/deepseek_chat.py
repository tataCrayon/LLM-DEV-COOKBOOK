import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

# RAG 示例所需
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # 此处以 OpenAI Embeddings 为例
from langchain_community.vectorstores import FAISS

# Agent 示例所需
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType


# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
load_dotenv()

# 检查API密钥是否已设置
def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
    return key

def get_serpapi_key():
    key = os.getenv('SERPAPI_API_KEY')
    if key is None:
        raise ValueError("SERPAPI_API_KEY not found in environment variables.")
    return key

# --- LLM 初始化 ---
def create_deepseek_llm():
    api_key = get_deepseek_key()
    if not api_key:
        raise ValueError("没有ds key")
    return ChatDeepSeek(
        model = "deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

def create_tools_for_agent(llm):
    # 加载 serpapi工具
    tools =load_tools(["serpapi"],llm= llm)
    # 如果搜索完想再计算一下可以这么写
    # tools = load_tools(['serpapi', 'llm-math'], llm=llm)
    # 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
    # tools=load_tools(["serpapi","python_repl"])
    return tools

def create_search_agent():
    deepseek = create_deepseek_llm()
    tools = create_tools_for_agent(deepseek)
    # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
    agent = initialize_agent(
        tools,
        deepseek, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    return agent

def test_simple_llm_call():
    llm = create_deepseek_llm()
    # 封装消息
    # LangChain 支持的消息类型如下：
    # - 'human': 人类消息
    # - 'user': 用户消息
    # - 'ai': AI 消息
    # - 'assistant': 助手消息
    # - 'function': 函数消息
    # - 'tool': 工具消息
    # - 'system': 系统消息
    # - 'developer': 开发者消息
    messages = [
        ("system","您是一个LangChain入门大师，代码助手"),
        ("human","你好，langchain集成deepseek。请问怎么联网搜索？")
    ]
    
    for chunk in llm.stream(messages):
        print(chunk.text(), end="")

def test_agent_with_internet_search():
    """
    测试llm联网搜索
    """
    print("测试 Agent 进行网络搜索")
    messages = [
        ("system","您是一个LangChain入门大师，代码助手"),
        ("human","介绍一下长沙大模型岗位要求、技术站")
    ]
    agent = create_search_agent()
    agent.invoke(messages)
    

if __name__ == '__main__':
    test_agent_with_internet_search()