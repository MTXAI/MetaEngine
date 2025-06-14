# 基于 langchain 的 agent 应用, 支持记忆与反思
# llm, prompt, agent, memory


## LLM
from openai import OpenAI


client = OpenAI(
    # one api 生成的令牌
    api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
    base_url="http://localhost:3000/v1"
)
# chat_completion = client.chat.completions.create(
#     # model="doubao-1.5-lite-32k",
#     model="qwen-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": "假设你正在带货一本英语单词书, 请简短准确且友好礼貌地回复弹幕问题: 怎么发货?",
#         }
#     ]
# )
# print(chat_completion.choices[0].message.content)


## Memory
from typing import Any, Dict, List

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import AIMessage, BaseMessage, HumanMessage, get_buffer_string
from langchain.schema.language_model import BaseLanguageModel

from engine.human.agent.db import filter_message


class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed

        messages = filter_message(
            conversation_id=self.conversation_id, limit=self.message_limit
        )
        # 返回的记录按时间倒序，转为正序
        messages = list(reversed(messages))
        chat_messages: List[BaseMessage] = []
        for message in messages:
            chat_messages.append(HumanMessage(content=message["query"]))
            chat_messages.append(AIMessage(content=message["response"]))

        if not chat_messages:
            return []

        # prune the chat message if it exceeds the max token limit
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(
                    get_buffer_string(chat_messages)
                )

        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass


## Prompt



## Agent


