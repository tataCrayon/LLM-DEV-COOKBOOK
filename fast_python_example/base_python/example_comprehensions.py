"""
python推导式
"""
reviews = [
    "这个产品太棒了！",
    "不好用。",
    "强烈推荐，物超所值！",
    "有点小贵",
    "物流很快"
]

result = [review for review in reviews if len(review) > 5]
print(result)
