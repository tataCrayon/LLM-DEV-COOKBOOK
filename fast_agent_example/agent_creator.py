import os
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor


# 从我们自己的模块中导入工具列表
from .tools.web_tools import web_tools_list
# 从llms获取llm
from .llms.llm_clients import deepseek_llm
from .configs.prompt_config import REACT_PROMPT_TEMPLATE  # <--- 从配置导入模板


def create_agent_executor():
    """创建并返回一个配置好的Agent执行器"""
    
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # 2. 初始化LLM
    
    llm = deepseek_llm()

    # 3. 创建Agent
    tools = web_tools_list
    agent = create_react_agent(llm, tools, prompt)

    # 4. 创建Agent执行器
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return executor