import os

# 假设我们已经有了 Document 对象，或者直接使用字符串
from langchain_core.documents import Document

# --- 文本分割器 ---
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter, # 专门用于 Markdown
    TokenTextSplitter # 基于 Token 数量分割 (通常需要 LLM 的 tokenizer)
)
# 对于 TokenTextSplitter，你可能需要一个 tokenizer，比如 tiktoken (OpenAI 使用)
# pip install tiktoken

# 我们可以复用前面 Document Loader 示例中加载的文档
# 或者直接创建一个长字符串作为示例

# (从之前的示例中复制一些辅助函数，如果需要的话)
# def get_deepseek_key(): ...
# def create_deepseek_llm(): ...

# --- 示例用长文本 ---
long_text_example = """LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它使得应用程序能够：
- 具有上下文感知能力：将语言模型连接到上下文来源（提示指令、少量示例、需要响应的内容等）。
- 具有推理能力：依赖语言模型进行推理（关于如何根据提供的上下文进行操作、何时进行操作等）。

LangChain 的主要价值主张是：
1. 组件化：LangChain 提供了模块化的抽象，用于构建语言模型应用程序所需的组件。这些组件具有易于使用的实现，并且可以组合使用以创建复杂的应用程序。
2. 用例驱动的链：LangChain 还提供了特定用例的链，这些链是预先构建的组件组合，用于完成常见的语言模型任务。

文本分割是处理长文本时的重要步骤。
首先，你需要将文本加载到 Document 对象中。
然后，你可以选择一个文本分割器。
RecursiveCharacterTextSplitter 是一种常用的分割器，它会尝试按一系列字符（如 "\\n\\n", "\\n", " ", ""）递归地分割文本，直到块达到所需的大小。
CharacterTextSplitter 则更简单，它按指定的单个字符分割。
还有针对特定格式（如 Markdown）或基于 Token 数量的分割器。
选择合适的分割器和参数（如块大小 chunk_size 和块重叠 chunk_overlap）对于后续的嵌入和检索效果至关重要。
块太小可能丢失上下文，块太大可能超出模型限制或包含太多不相关信息。
重叠（overlap）有助于在块之间保持一定的上下文连续性。
"""

# 创建一个 Document 对象用于测试
sample_document = Document(page_content=long_text_example, metadata={"source": "manual_example"})

# --- 示例 1: CharacterTextSplitter ---
def split_with_character_splitter(doc_to_split):
    print("\n--- 示例 1: CharacterTextSplitter ---")
    # CharacterTextSplitter 按指定的单个字符进行分割。
    # 如果不指定 separator，它会尝试按 "\n\n" 分割，如果块仍然太大，则会报错或行为不如预期。
    # 通常我们会明确指定一个简单的分隔符。
    text_splitter = CharacterTextSplitter(
        separator="\n", # 按换行符分割
        chunk_size=200,  # 每个块的最大字符数 (大致)
        chunk_overlap=20, # 块之间的重叠字符数，以保持上下文连贯
        length_function=len, # 用于计算长度的函数，默认为 len
        is_separator_regex=False, # separator 不是正则表达式
    )

    chunks = text_splitter.split_documents([doc_to_split])
    # 也可以用 text_splitter.split_text(doc_to_split.page_content) 来分割纯文本

    print(f"原始文档字符数: {len(doc_to_split.page_content)}")
    print(f"分割成了 {len(chunks)} 个块。")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1} (字符数: {len(chunk.page_content)}):")
        print(f"  内容 (前80字符): '{chunk.page_content[:80]}...'")
        print(f"  元数据: {chunk.metadata}") # 元数据会从原始文档继承VerctorStores
    return chunks

# --- 示例 2: RecursiveCharacterTextSplitter ---
def split_with_recursive_character_splitter(doc_to_split):
    print("\n--- 示例 2: RecursiveCharacterTextSplitter ---")
    # 这是推荐的通用文本分割器。
    # 它会按一个字符列表递归地尝试分割，直到块达到期望的大小。
    # 默认的分隔符列表是 ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,  # 每个块的目标大小 (字符数)
        chunk_overlap=30,  # 块之间的重叠字符数
        length_function=len,
        # separators=None, # 可以自定义分隔符列表，默认为 ["\n\n", "\n", " ", ""]
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents([doc_to_split])

    print(f"原始文档字符数: {len(doc_to_split.page_content)}")
    print(f"分割成了 {len(chunks)} 个块。")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1} (字符数: {len(chunk.page_content)}):")
        print(f"  内容 (前80字符): '{chunk.page_content[:80]}...'")
        print(f"  元数据: {chunk.metadata}")
    return chunks

# --- 示例 3: MarkdownHeaderTextSplitter (特定格式) ---
def split_with_markdown_splitter():
    print("\n--- 示例 3: MarkdownHeaderTextSplitter ---")
    markdown_text = """
# LangChain 简介

## 核心组件

LangChain 提供了多种核心组件来构建 LLM 应用。

### 模型 I/O

包括 LLMs, Chat Models, Prompts, Output Parsers。

### 数据连接

包括 Document Loaders, Text Splitters, Embeddings, Vector Stores。

## 链 (Chains)

链是将组件按顺序组合起来的方式。

## 代理 (Agents)

代理使用 LLM 来决定采取哪些行动。
"""
    # 定义 Markdown 的标题层级和对应的元数据字段名
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # split_text 返回的是 Document 对象列表，每个对象的元数据会包含它所属的标题
    md_chunks = markdown_splitter.split_text(markdown_text)

    print(f"Markdown 文本分割成了 {len(md_chunks)} 个块。")
    for i, chunk in enumerate(md_chunks):
        print(f"\n块 {i+1}:")
        print(f"  内容 (前80字符): '{chunk.page_content[:80]}...'")
        print(f"  元数据: {chunk.metadata}") # 元数据会包含 Header 1, Header 2 等
    return md_chunks

# --- 示例 4: TokenTextSplitter (基于 Token) ---
# 需要 tiktoken: pip install tiktoken
# 需要一个 LLM 模型名称来确定 tokenizer，或者直接提供 tokenizer
def split_with_token_splitter(text_to_split, model_name="gpt-3.5-turbo"):
    print("\n--- 示例 4: TokenTextSplitter ---")
    try:
        text_splitter = TokenTextSplitter(
            model_name=model_name, # 或者直接传递 encoding_name="cl100k_base" (GPT-3.5/4)
            chunk_size=100,  # 每个块的目标 Token 数
            chunk_overlap=10,  # 块之间的重叠 Token 数
        )
        # TokenTextSplitter 通常直接操作文本字符串
        chunks_text = text_splitter.split_text(text_to_split)

        print(f"原始文本字符数: {len(text_to_split)}")
        print(f"分割成了 {len(chunks_text)} 个文本块 (基于 Token)。")
        for i, chunk_str in enumerate(chunks_text):
            # 你可以手动计算 token 数来验证 (近似)
            # import tiktoken
            # enc = tiktoken.encoding_for_model(model_name)
            # num_tokens = len(enc.encode(chunk_str))
            print(f"\n文本块 {i+1} (字符数: {len(chunk_str)}):") # , Token 数 (近似): {num_tokens}
            print(f"  内容 (前80字符): '{chunk_str[:80]}...'")
        return chunks_text
    except ImportError:
        print("请安装 'tiktoken' 库以使用 TokenTextSplitter: pip install tiktoken")
        return []
    except Exception as e:
        print(f"TokenTextSplitter 初始化或使用时出错: {e}")
        print("确保你指定的 model_name 是 tiktoken 支持的，或者 tiktoken 已正确安装。")
        return []


if __name__ == '__main__':
    # 使用之前定义的 sample_document
    print(f"原始示例文档内容:\n'''{sample_document.page_content}'''")

    char_chunks = split_with_character_splitter(sample_document)
    recursive_chunks = split_with_recursive_character_splitter(sample_document)
    md_chunks = split_with_markdown_splitter() # 这个用的是独立的 markdown 文本
    token_chunks_text = split_with_token_splitter(sample_document.page_content)

    print("\n--- 分割器对比总结 ---")
    print(f"CharacterTextSplitter 块数: {len(char_chunks)}")
    print(f"RecursiveCharacterTextSplitter 块数: {len(recursive_chunks)}")
    print(f"MarkdownHeaderTextSplitter 块数: {len(md_chunks)}")
    print(f"TokenTextSplitter 块数: {len(token_chunks_text)}")

    # 在实际应用中，你会选择一种分割器，然后将分割后的块 (chunks)
    # 用于下一步，比如创建 Embeddings 并存入 VectorStore。
    # 例如，如果选择了 recursive_chunks:
    # relevant_chunks_for_rag = recursive_chunks
    # print(f"\n准备将 {len(relevant_chunks_for_rag)} 个块用于 RAG...")