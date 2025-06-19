
# 直接使用serp-api

from langchain.utilities import SerpAPIWrapper
# Initialize the SerpAPI wrapper
search = SerpAPIWrapper()
# Perform a search
results = search.run("What is LangChain?")
print(results)


# 在LangChain Agent中使用

