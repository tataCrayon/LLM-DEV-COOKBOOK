import os
from dotenv import load_dotenv
# 基础对话所需
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 提示模板 - ConversationChain 有自己的默认模板，但我们也可以自定义
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,  # 非常重要，用于在提示中为历史消息占位
    HumanMessagePromptTemplate
)
# Memory 组件
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory

# 链 - 我们会使用 ConversationChain
from langchain.chains import ConversationChain


# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
load_dotenv()


def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None:
        raise ValueError(
            "DEEPSEEK_API_KEY not found in environment variables.")
    return key

# --- LLM 初始化 ---


def create_deepseek_llm():
    api_key = get_deepseek_key()
    if not api_key:
        raise ValueError("没有ds key")
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.1,  # 温度，更具备确定性
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

# --- 带记忆的对话测试 ---


def test_conversation_with_buffer_memory():
    print("使用ConversationBufferMemory测试带记忆的对话")
    llm = create_deepseek_llm()

    """
    1.初始化 Memory
    `memory_key="history"` 是 ConversationBufferMemory 默认存储历史记录的变量名。
    `return_messages=True` 表示 memory 将以消息对象列表的形式返回历史记录，这对于 ChatModels 更好。
    """
    memory = ConversationBufferMemory(
        memory_key="history", return_messages=True)

    """
    2. 为ConversationChain自定义一个提示模板
    ConversationChain 有一个默认的提示模板，但我们可以自定义以加入系统消息等。
    使用MessagesPlaceholder，它告诉模板在哪里插入历史消息
    `input` 是 ConversationChain 期望的用户输入变量名。
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage("你是一个乐于助人的LangChain专家，擅长简洁明了地解释概念。"),
            MessagesPlaceholder(variable_name="history"),  # Memory中的历史消息会插入这里
            HumanMessagePromptTemplate.from_template("{input}")  # 用户输入会插入这里
        ]
    )

    """
    3. 初始化 ConversationChain
    verbose=True 表示 ConversationChain 会打印出每个消息的处理结果。
    可以让我们看到链的执行过程，包括发送给 LLM 的完整提示
    """
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template,# 使用自定义的提示模板
        verbose=True
    )
    
    # --- 开始多轮对话 ---
    print("\n 第一轮对话")
    response1 = conversation_chain.invoke({"input": "你好，我是啾啾。请问LangChain中的Memory组件是做什么用的？"})
    # response1 是一个字典，通常包含 'input', 'history', 'response' (LLM的回答)
    print(f"AI 回答: {response1['response']}")

    print("\n--- 查看当前 Memory 内容 ---")
    # load_memory_variables({}) 会加载所有存储的变量，这里主要是 history
    print(memory.load_memory_variables({}))


    print("\n第二轮对话:")
    response2 = conversation_chain.invoke({"input": "很好，那我刚才告诉你我的名字了吗？如果说了，是什么？"})
    print(f"AI 回答: {response2['response']}")

    print("\n--- 再次查看当前 Memory 内容 (应该包含第一轮和第二轮) ---")
    print(memory.load_memory_variables({}))

    print("\n第三轮对话 (测试 AI 是否还记得我叫啾啾):")
    response3 = conversation_chain.invoke({"input": "你觉得我提出的关于Memory组件的问题怎么样？"}) 
    print(f"AI 回答: {response3['response']}")
    
    print("\n--- 测试结束 ConversationBufferMemory ---")
    

# --- 另一种 Memory 类型：`ConversationBufferWindowMemory` ---
# 这种 Memory 只会保留最近的 K 轮对话，防止上下文过长超出 LLM 限制或消耗过多 token。
def test_conversation_with_window_memory():
    print("\n\n--- 开始测试 ConversationBufferWindowMemory (k=2) ---")
    llm = create_deepseek_llm()

    # 1. 初始化 Window Memory，k=2 表示只保留最近2轮对话
    # (Human输入 + AI输出 算一轮交互，但这里k指的是交互的“对数”或消息数，取决于具体实现，通常是交互对)
    # 为了清晰，我们说 k=2 保留最近2次完整的 "Human: ..." 和 "AI: ..." 交换
    window_memory = ConversationBufferWindowMemory(k=2, memory_key="history", return_messages=True)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个简明扼要的AI助手。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation_chain_window = ConversationChain(
        llm=llm,
        memory=window_memory,
        prompt=prompt_template,
        verbose=True
    )

    print("\n第1轮:")
    conversation_chain_window.invoke({"input": "我喜欢蓝色。"})
    print(f"Memory: {window_memory.load_memory_variables({})}")

    print("\n第2轮:")
    conversation_chain_window.invoke({"input": "我最喜欢的食物是披萨。"})
    print(f"Memory: {window_memory.load_memory_variables({})}") # 应该包含蓝色和披萨

    print("\n第3轮:")
    conversation_chain_window.invoke({"input": "我住在北京。"})
    # 因为 k=2, "我喜欢蓝色" 这一轮对话应该被挤掉了
    print(f"Memory: {window_memory.load_memory_variables({})}") # 应该只包含披萨和北京

    print("\n第4轮 (测试AI是否还记得第一轮内容):")
    response = conversation_chain_window.invoke({"input": "我最开始说的我喜欢什么颜色？"})
    print(f"AI 回答: {response['response']}") # AI 可能不记得了，或猜一个

    print("\n--- 测试结束 ConversationBufferWindowMemory ---")
    
if __name__ == '__main__':
    # test_conversation_with_buffer_memory()
    test_conversation_with_window_memory()