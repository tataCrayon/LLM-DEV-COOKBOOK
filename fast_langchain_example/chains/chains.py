import os
from dotenv import load_dotenv

# 基础对话所需
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 提示模板
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# 输出解析器
from langchain_core.output_parsers import StrOutputParser

# 链条
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# 用于传递原始输入或修改字典
from langchain_core.runnables import RunnablePassthrough



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
        temperature=1,  # 低温度，更具备确定性
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )


def test_simple_sequential_chain():
    print("测试简单序列链条")
    # step 1:创建生成故事标题的链实例1
    deepseek = create_deepseek_llm()
    title_prompt = ChatPromptTemplate.from_template(
        "你是一位富有想象力的作家。为一个关于“{topic}”的短篇故事写一个引人入胜的标题。"
    )
    title_chain = title_prompt | deepseek | StrOutputParser()  # 使用LCEL

    # step 2:创建基于标题写故事情节的链实例2
    synopsis_prompt = ChatPromptTemplate.from_template(
        "你是一位编剧。根据以下标题写一个简短的故事情节（大约100字）：\n标题：{story_title}"
    )
    """
    SimpleSequentialChain 期望每个链只有一个输入（在第一个链中）和一个输出
    它会自动将第一个链的输出传递给第二个链的输入（变量名需要匹配或自动推断）
    如果前一个链的输出是字符串，且后一个链的提示模板只有一个输入变量，它会自动映射
    """
    synopsis_chain = synopsis_prompt | deepseek | StrOutputParser()

    """
    setp 3:在链之间传递值 使用RunnablePassthrough.assign()
    
    """
    full_chain = RunnablePassthrough.assign(
        story_title_from_llm=title_chain
    ) | {"story_title": lambda x: x["story_title_from_llm"], } | synopsis_chain

    # step 4:运行链
    topic_input = "一个能与动物对话的女孩"
    try:
        print(f"\n为主题“{topic_input}”生成故事:")
        synopsis = full_chain.invoke({"topic": "一个能与动物对话的女孩"})
        print(f"生成的故事概要: {synopsis}")
    except Exception as e:
        import traceback
        print(f"\n生成故事失败: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    test_simple_sequential_chain()
