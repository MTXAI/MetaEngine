import asyncio
from fastapi import Request

from engine.human.agent.chat.kb_chat import kb_chat

# 构造参数
query = "你好"
mode = ["local_kb"]
kb_name = "samples"
top_k = 5
score_threshold = 0.5
history = [
    {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
    {"role": "assistant", "content": "虎头虎脑"}
]
stream = True
model = "your-llm-model"
temperature = 0.7
max_tokens = 1024
prompt_name = "default"
return_direct = False
request = Request(scope={"type": "http"})

# 假设 kb_chat 已经被导入
# 有错 pydantic.errors.PydanticUserError: The `__modify_schema__` method is not supported in Pydantic v2. Use `__get_pydantic_json_schema__` instead in class `SecretStr`.
result = asyncio.run(
    kb_chat(
        query=query,
        mode=mode,
        kb_name=kb_name,
        top_k=top_k,
        score_threshold=score_threshold,
        history=history,
        stream=stream,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_name=prompt_name,
        return_direct=return_direct,
        request=request
    )
)
print(result)