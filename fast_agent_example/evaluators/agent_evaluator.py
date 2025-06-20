from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from ..llms.llm_clients import deepseek_llm # 从我们的llm模块导入

from ..configs.prompt_config import EVALUATION_PROMPT_TEMPLATE

def evaluate_agent_trace(question: str, agent_trace: str, final_answer: str):
    """
    使用LLM作为评委，来评估Agent的执行轨迹和最终答案。
    """
    print("\n--- 启动LLM评委进行评估 ---")
    
    # 1. 创建评估链 (Evaluation Chain)
    prompt = PromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
    llm = deepseek_llm() # 复用我们创建LLM的函数
    
    # 使用LCEL（LangChain Expression Language）来构建链
    # 这是一个简单的 "Prompt -> LLM -> String Output" 链
    evaluation_chain = prompt | llm | StrOutputParser()
    
    # 2. 运行评估链
    try:
        print("评委正在审阅材料并生成报告...")
        evaluation_report = evaluation_chain.invoke({
            "question": question,
            "agent_trace": agent_trace,
            "final_answer": final_answer
        })
        
        print("\n--- 评委报告生成完毕 ---")
        print(evaluation_report)
        
    except Exception as e:
        print(f"\n--- LLM评委在评估时发生错误 ---")
        print(e)