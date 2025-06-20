from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def search_tool(query: str) -> str:
    """
    当需要在线搜索最新信息、查找文章链接或回答关于时事的问题时，使用此工具。
    输入应该是一个精确的搜索查询语句。
    """
    print(f"\n>> 调用搜索工具，查询：'{query}'")
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool
def scrape_website_tool(url: str) -> str:
    """
    当需要从一个给定的网页URL中获取详细文本内容时，使用此工具。
    输入必须是一个有效的URL。
    """
    print(f"\n>> 调用网页抓取工具，URL：'{url}'")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:4000]
    except Exception as e:
        return f"抓取失败: {e}"

# 导出一个包含所有工具的列表，方便其他模块导入
web_tools_list = [search_tool, scrape_website_tool]