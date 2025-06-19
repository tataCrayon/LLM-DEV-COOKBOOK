import os
from dotenv import load_dotenv
# 基础对话所需
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 提示模板
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 环境key
load_dotenv()

def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
    return key

# --- LLM 初始化 ---
def create_deepseek_llm():
    api_key = get_deepseek_key()
    if not api_key:
        raise ValueError("没有ds key")
    return ChatDeepSeek(
        model = "deepseek-chat",
        temperature=0.1, # 低温度，更具备确定性
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

 
# --- 使用模板对话限定输入输出 ---
def create_langchain_teacher_template():
    # 创建基础对话模板 
    """ 
    我们能在一些消息的构建上看到这种使用消息对象的写法。
    但是实际上这不是推荐的简洁写法，应该直接使用元组。
    
    是的，这个写法不会被视为一个s-string模板，而是一个string
    return ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个乐于助人的LangChain专家。"),
            HumanMessage(content="你好，我想问一个关于LangChain的问题: {actual_user_input}")
    ])
    """
    return ChatPromptTemplate.from_messages([
            ("system", "你是一个乐于助人的LangChain专家。"),
            ("human", "你好，我想问一个关于LangChain的问题: {actual_user_input}") # 确保这里是占位符
    ])

def test_simple_template_call():
    llm = create_deepseek_llm()
    user_question = "你好，请问LangChain组件Prompt Templates可以做什么？"
    prompt_template = create_langchain_teacher_template()
    messages = prompt_template.format_messages(actual_user_input=user_question)
    
    print("最终发送给 LLM 的消息:")
    for msg in messages:
        print(f"- 类型: {msg.type}, 内容: {msg.content}")
    
    
    print("\nAI开始回答")
    for chunk in llm.stream(messages):
        print(chunk.content, end="")
    print("\nAI回答完了")
    

if __name__ == '__main__':
    test_simple_template_call()