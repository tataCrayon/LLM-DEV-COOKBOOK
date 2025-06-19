import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 提示模板
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
load_dotenv()

def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None:
        raise ValueError(
            "DEEPSEEK_API_KEY 未在环境变量中找到。")
    return key

# --- LLM 初始化 ---


def create_deepseek_llm():
    api_key = get_deepseek_key()
    # get_deepseek_key 已经检查了 key 是否存在，这里无需重复检查
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.1,  # 低温度，更具确定性
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

# --- 创建带上下文的模板 ---
def create_langchain_teacher_template():
    return ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的LangChain专家，基于对话历史提供准确、简洁的回答。"),
        MessagesPlaceholder(variable_name="history"), # optional=True 也可以，但如果历史总是存在，则非必需
        ("human", "你好，我想问一个关于LangChain的问题: {actual_user_input}")
    ])

# --- 测试带上下文的对话 ---
def test_context_aware_template_call():
    llm = create_deepseek_llm()
    prompt_template = create_langchain_teacher_template()

    # 初始化记忆机制
    # 告诉记忆模块人类输入对应的键名是什么
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",  # 这是默认值，与 MessagesPlaceholder 匹配
        input_key="actual_user_input"  # 关键：告诉记忆模块当前人类输入对应的键名
    )

    # 告诉 ConversationChain 它的主要输入键是什么
    chain = ConversationChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        input_key="actual_user_input",  # 关键：告诉链，run()方法中的哪个参数是主要的用户输入
        verbose=True # 开启详细模式，方便调试，会打印完整的提示
    )

    # 第一次提问
    user_question_1 = "你好，请问LangChain组件Prompt Templates可以做什么？"
    print(f"\n用户提问: {user_question_1}")
    # 当调用 run 时，使用在 ConversationChain 的 input_key 中指定的键
    response_1 = chain.run(actual_user_input=user_question_1)
    print("AI回答:", response_1)

    # 第二次提问，依赖上下文
    user_question_2 = "那它支持多轮对话吗？"
    print(f"\n用户提问: {user_question_2}")
    response_2 = chain.run(actual_user_input=user_question_2)
    print("AI回答:", response_2)

    # 你可以检查记忆中的内容
    print("\n--- 记忆内容 ---")
    print(memory.load_memory_variables({}))


if __name__ == '__main__':
    test_context_aware_template_call()