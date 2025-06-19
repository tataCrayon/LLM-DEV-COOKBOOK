from langchain.text_splitter import RecursiveCharacterTextSplitter

text_to_split = """
RAG的核心思想是开卷考试。您可以把传统的LLM想象成一个“闭卷考试”，它只能根据自己脑海里预先训练好的知识来回答问题。如果问它一个最新的、或者私有的信息，它就会说“我不知道”。

而RAG (Retrieval-Augmented Generation)，就是把这个过程变成了一场“开卷考试”。在回答你的问题之前，它会先去一个我们指定的“书架”（也就是你的知识库）上，快速“检索”到最相关的几页“书”（也就是文本片段），然后把这些内容和你的问题一起，作为参考资料交给LLM，让它“阅读并总结”出答案。
"""

# --- 核心修改在这里 ---
# 1. 预处理文本：将所有换行符替换为空格
processed_text = text_to_split.replace("\n", " ")

# 2. 创建分割器实例 (参数不变)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# 3. 使用预处理后的文本进行分割
#    注意：create_documents需要一个列表，所以我们传入 [processed_text]
chunks = text_splitter.create_documents([processed_text])

# 4. 打印结果
for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(chunk.page_content)
    print(f"长度: {len(chunk.page_content)}\n")