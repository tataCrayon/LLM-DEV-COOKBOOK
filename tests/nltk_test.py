import nltk
nltk.data.path.append(r'F:\DependencyPackages\llmRepository\nltk_data')
from langchain.text_splitter import TextSplitter

# 我们的原始文本
text_to_split = """
RAG的核心思想是开卷考试。您可以把传统的LLM想象成一个“闭卷考试”，它只能根据自己脑海里预先训练好的知识来回答问题。如果问它一个最新的、或者私有的信息，它就会说“我不知道”。

而RAG (Retrieval-Augmented Generation)，就是把这个过程变成了一场“开卷考试”。在回答你的问题之前，它会先去一个我们指定的“书架”（也就是你的知识库）上，快速“检索”到最相关的几页“书”（也就是文本片段），然后把这些内容和你的问题一起，作为参考资料交给LLM，让它“阅读并总结”出答案。
"""

# 1. 先用NLTK将文本分割成句子
sentences = nltk.sent_tokenize(text_to_split)

# 2. 现在我们手动将句子组合成块，同时考虑chunk_size和overlap
chunk_size = 100
chunk_overlap = 20
chunks = []
current_chunk = ""

for sentence in sentences:
    # 如果把新句子加进来，当前块会超过大小限制
    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
        chunks.append(current_chunk)
        # 创建重叠部分
        overlap_part = " ".join(current_chunk.split()[-chunk_overlap//2:]) # 取最后几个词作为重叠
        current_chunk = overlap_part + " " + sentence
    else:
        current_chunk += " " + sentence

# 加入最后一个块
if current_chunk:
    chunks.append(current_chunk.strip())

# 3. 打印结果
for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(chunk)
    print(f"长度: {len(chunk)}\n")