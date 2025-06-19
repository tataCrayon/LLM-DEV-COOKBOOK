import pandas as pd # "pd" 是使用Pandas的标准别名

available_models_list = [
    {"name": "gpt-4-turbo", "context_window": 128000, "is_multimodal": True},
    {"name": "claude-3-opus", "context_window": 200000, "is_multimodal": True},
    {"name": "gemini-1.5-pro", "context_window": 1000000, "is_multimodal": True},
    {"name": "text-embedding-ada-002", "context_window": 8191, "is_multimodal": False}
]

# 只需一行代码，就能创建DataFrame
df = pd.DataFrame(available_models_list)

# 当你打印或显示df时，会看到一个漂亮的表格：
#
#                      name  context_window  is_multimodal
# 0             gpt-4-turbo          128000           True
# 1           claude-3-opus          200000           True
# 2          gemini-1.5-pro         1000000           True
# 3  text-embedding-ada-002            8191          False

"""
数据立刻变得清晰、结构化了。左边的 `0, 1, 2, 3` 就是索引，
顶部的 `name, context_window, is_multimodal` 就是列。
DataFrame的超能力：
轻松选取一整列：想看所有的模型名称？
可以直接用 `df['name']` 选取 `name` 列。
"""
print(df['name'])
print("\n")
# 如果想对列数据进行筛选，如选择那些'is_multimodal'列的值为True的行
multimodal_models = df[df['is_multimodal'] == True]
print(multimodal_models)
print("\n")

# 快速统计分析：想知道最大上下文窗口是多少？
max_context = df['context_window'].max()
print(max_context) # 输出: 1000000

print("\n")

# 只用一行Pandas代码，筛选出并显示所有 `context_window` 大于 150,000 的模型。

windows = df[df['context_window'] > 150000]
print(windows)  