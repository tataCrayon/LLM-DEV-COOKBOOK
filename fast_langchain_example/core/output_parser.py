import os
from dotenv import load_dotenv

# 基础对话所需
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

# 提示模板
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 输出解析器
from langchain.output_parsers import CommaSeparatedListOutputParser

# 链条
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
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
    return ChatPromptTemplate.from_messages([
            ("system", "你是一个乐于助人的LangChain专家。"),
            ("human", "你好，我想问一个关于LangChain的问题: {actual_user_input} \n\n {format_instructions}") # 确保这里是占位符
    ])


# --- 使用输出解析器解析输出 ---
def create_output_parser():
    return CommaSeparatedListOutputParser()

def test_output_parser_with_template():
    llm = create_deepseek_llm()
    output_parser = create_output_parser()
    """
    1.获取格式化指令
    对于 CommaSeparatedListOutputParser，它会是类似 "Your response should be a list of comma separated values, eg: `foo, bar, baz`"
    """
    format_instructions = output_parser.get_format_instructions()
    print(f"输出解析器的格式化指令: {format_instructions}")
    
    prompt_template = create_langchain_teacher_template()
    user_task = "请列出 LangChain 的3个主要优点。"
    
    # 2.格式化完整提示，包含用户任务和格式指令
    messages_for_llm = prompt_template.format_messages(
        actual_user_input=user_task,
        format_instructions=format_instructions
    )
    
    print("\n最终发送给 LLM 的消息 (包含格式指令):")
    for msg in messages_for_llm:
        print(f"- 类型: {msg.type}, 内容: {msg.content}")
    
    # 3. 调用LLM并获取原始文本输出
    print("\nAI开始生成原始文本 (等待 LLM 响应)...")
    ai_response_message = llm.invoke(messages_for_llm)
    raw_llm_output = ai_response_message.content # AIMessage对象的content属性是字符串
    print(f"LLM 返回的原始文本: '{raw_llm_output}'")
    
    # 4. 解析原始文本输出
    try:
        parsed_output = output_parser.parse(raw_llm_output)
        print("\n解析后的输出 (Python列表):")
        print(parsed_output)
        if isinstance(parsed_output, list):
            print("LangChain 的3个优点是:")
            for i,advantage in enumerate(parsed_output):
                print(f"{i}. {advantage.strip()}")
    except Exception as e:
        print(f"解析输出时出错: {e}")
        print("这通常意味着 LLM 的输出没有严格遵循格式化指令。")
        print("你可以尝试调整提示，或者使用更鲁棒的解析器/重试机制。")
    
    print("--- 结束测试：带输出解析器的模板调用 ---\n")

        
    

if __name__ == '__main__':
    test_output_parser_with_template()