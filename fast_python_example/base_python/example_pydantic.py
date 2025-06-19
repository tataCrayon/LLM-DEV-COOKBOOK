from pydantic import BaseModel
from typing import List, Dict

# 1. 我们用类型提示来“定义”我们期望的数据结构
class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class APIResponse(BaseModel):
    id: str
    model: str
    choices: List[Choice] # 这里明确指出choices是一个包含Choice对象的列表
    usage: Dict[str, int]

def get_from_api():
    api_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好！有什么可以帮助你的吗？"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    return api_response

# 2. 我们用这个定义好的结构来“解析”原始数据
raw_data = get_from_api() # 假设这里拿到了上面的字典数据
response_model = APIResponse.model_validate(raw_data)

# 3. 输入 response_model. 就会看到 id, model, choices...类型是安全的
content = response_model.choices[0].message.content 