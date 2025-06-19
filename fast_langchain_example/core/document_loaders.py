import os
from dotenv import load_dotenv

# LLM 和 LangChain 基础 (保留上下文，尽管在加载器示例中不直接使用)
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain  # 如果我们想使用加载的文档，可以用于一个简单的链
from langchain.memory import ConversationBufferMemory

# --- 文档加载器 ---
from langchain_community.document_loaders import TextLoader  # 用于 .txt 文件
from langchain_community.document_loaders import PyPDFLoader  # 用于 .pdf 文件
from langchain_community.document_loaders import WebBaseLoader  # 用于加载网页



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


# --- 示例 1: TextLoader ---
def load_from_text_file():
    print("\n--- 示例 1: 从 .txt 文件加载 ---")
    # 为此示例创建一个虚拟文本文件
    file_path = "my_sample_document.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("这是我的示例文档的第一行。\n")
        f.write("LangChain的文档加载器非常有用。\n")
        f.write("它可以帮助我们从不同来源加载数据。\n")
        f.write("比如文本文件、PDF，甚至是网页。\n")

    # 使用文件路径初始化 TextLoader
    loader = TextLoader(file_path, encoding="utf-8")  # 如果不是默认编码，请指定

    # 加载文档
    # .load() 方法返回一个 Document 对象列表。
    # 对于 TextLoader，它通常为整个文件返回单个 Document 对象。
    documents = loader.load()

    print(f"加载了 {len(documents)} 个文档。")
    for i, doc in enumerate(documents):
        print(f"\n文档 {i+1}:")
        print(f"  内容 (前 100 个字符): {doc.page_content[:100]}...")
        print(f"  元数据: {doc.metadata}")  # 元数据将包含源文件路径

    # 清理虚拟文件
    os.remove(file_path)
    return documents  # 返回以便将来使用

# --- 示例 2: PyPDFLoader ---


def load_from_pdf_file():
    print("\n--- 示例 2: 从 .pdf 文件加载 ---")
    # 对于此示例，你需要一个 PDF 文件。
    # 假设你在同一目录中有一个名为 "sample.pdf" 的 PDF 文件。
    # 如果你没有，可以在线下载一个示例 PDF。
    # 例如，搜索 "dummy pdf" 或 "sample pdf"。
    pdf_file_path = "sample.pdf"  # 替换为你的 PDF 文件路径

    if not os.path.exists(pdf_file_path):
        print(f"PDF 文件 '{pdf_file_path}' 未找到。跳过 PyPDFLoader 示例。")
        print("请创建或下载一个 'sample.pdf' 文件并将其放置在脚本的目录中。")
        return []

    # 初始化 PyPDFLoader
    loader = PyPDFLoader(pdf_file_path)

    # 加载文档
    # PyPDFLoader 通常为 PDF 中的每一页创建一个 Document 对象。
    # 对于大型 PDF，.load() 可能会比较慢。
    # 也可以使用 .load_and_split()，它会先加载然后按页分割。
    pages = loader.load()  # 将每一页加载为一个单独的 Document

    print(f"从 PDF 加载了 {len(pages)} 页。")
    if pages:
        print(f"\n第 1 页内容 (前 150 个字符): {pages[0].page_content[:150]}...")
        print(f"第 1 页元数据: {pages[0].metadata}")  # 将包括来源和页码

        if len(pages) > 1:
            print(f"\n第 2 页内容 (前 150 个字符): {pages[1].page_content[:150]}...")
            print(f"第 2 页元数据: {pages[1].metadata}")
    return pages

# --- 示例 3: WebBaseLoader ---
# 需要: pip install beautifulsoup4


def load_from_webpage():
    print("\n--- 示例 3: 从网页加载 ---")
    # target_url = "https://blog.langchain.dev/langchain-expression-language/" # 一篇 LangChain 博客文章
    # 一篇关于 LLM Agent 的知名博客文章
    target_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    # 初始化 WebBaseLoader
    # 它使用 urllib 获取页面并使用 BeautifulSoup 解析 HTML。
    loader = WebBaseLoader(target_url)

    # 你也可以传递多个 URL:
    # loader = WebBaseLoader(["url1", "url2"])

    print(f"正在从 {target_url} 加载内容...")
    try:
        documents = loader.load()  # 返回一个列表 (通常每个 URL 一个 Document)
        print(f"从网页加载了 {len(documents)} 个文档。")
        if documents:
            print(
                f"\n文档 1 内容 (前 200 个字符): {documents[0].page_content[:200]}...")
            print(f"文档 1 元数据: {documents[0].metadata}")  # 包括来源 URL 和标题
        return documents
    except Exception as e:
        print(f"加载网页时出错: {e}")
        print("这可能是由于网络问题、网站结构或缺少依赖项（如 'html2text'）造成的。")
        print("如果你看到 'No module named html2text'，请运行: pip install html2text")
        return []


# --- 如何使用加载的文档 (非常基础的示例) ---
def use_loaded_document_in_chain(documents):
    if not documents:
        print("\n未加载任何文档，跳过链示例。")
        return

    # 为简单起见，我们只取第一个文档的内容
    # 在一个真正的 RAG 系统中，你会做更多的事情 (分割、嵌入、检索)
    context_text = documents[0].page_content

    llm = create_deepseek_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个仅根据提供的上下文回答问题的助手。如果答案不在上下文中，请说“根据提供的文本我不知道”。"),
        ("human", "上下文:\n{context}\n\n问题: {question}")
    ])

    chain = LLMChain(llm=llm, prompt=prompt)

    # 示例问题 - 这将高度依赖于你文档的内容
    # 对于 "my_sample_document.txt":
    # question = "文档加载器有什么用？"
    # 对于 PDF 或网页，你会问一个与其内容相关的问题。
    # 对于 Lilian Weng 博客文章示例:
    question = "一个由LLM驱动的自主代理系统的关键组成部分是什么？"

    print(f"\n--- 在简单链中使用加载的文档 ---")
    print(f"上下文 (前 100 个字符): {context_text[:100]}...")
    print(f"问题: {question}")

    # 流式传输响应
    response_content = ""
    print("\nAI 回答:")
    # chain.stream 会产生包含 'text' 键的字典（对于 LLMChain）
    for chunk in chain.stream({"context": context_text, "question": question}):
        if 'text' in chunk:
            print(chunk['text'], end="", flush=True)
            response_content += chunk['text']

    if not response_content:  # 如果流式传输没有产生预期的结构，则回退
        result = chain.invoke({"context": context_text, "question": question})
        print(result['text'])
    print("\n--- 链示例结束 ---")


if __name__ == '__main__':
    # 确保 DEEPSEEK_API_KEY 环境变量已设置或在代码中直接提供
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("警告：DEEPSEEK_API_KEY 环境变量未设置。LLM 调用可能会失败。")
        # 你可以在这里设置一个默认的 key，或者提示用户输入
        # os.environ["DEEPSEEK_API_KEY"] = "sk-your-actual-key"

    # 测试 TextLoader
    txt_docs = load_from_text_file()
    # 如果你加载了一个包含相关信息的文本文件，你可以将 txt_docs 传递给 use_loaded_document_in_chain

    # 测试 PyPDFLoader
    # 确保有一个 sample.pdf 文件或更改路径
    pdf_pages = load_from_pdf_file()
    # if pdf_pages:
    #     use_loaded_document_in_chain(pdf_pages) # 示例：问一个关于你的 PDF 的问题

    # 测试 WebBaseLoader
    web_docs = load_from_webpage()
    if web_docs:
        use_loaded_document_in_chain(web_docs)  # 示例：问一个关于网页内容的问题
