import numpy as np

# 假设这是“天气真好”的向量
vector_a = np.array([0.1, 0.5, 0.9])

# 假设这是“今天阳光明媚”的向量
vector_b = np.array([0.2, 0.6, 0.8])

# 方法一：使用dot函数
dot_product_1 = np.dot(vector_a, vector_b)

# 方法二：使用专用的矩阵乘法运算符（在现代代码中更受青睐）
dot_product_2 = vector_a @ vector_b

print(dot_product_1) # 输出: 1.04
print(dot_product_2) # 输出: 1.04