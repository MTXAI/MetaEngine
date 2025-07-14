from engine.human.character.agent.base_agent import BaseAgent
from engine.human.character.processor.processor import BaseProcessor

class Character:
    def __init__(self, agent_model: BaseAgent, agent_processor: BaseProcessor, agent_prompt: str = None):
        """
        agent_model: 代理的核心模型，需继承自BaseAgent
        agent_prompt: 角色的提示词或模板
        agent_processor: 输入输出处理器
        """
        self.agent_model = agent_model
        self.agent_processor = agent_processor
        self.agent_prompt = agent_prompt

    def answer(self, question: str, **kwargs) -> str:
        # 可在此处加入prompt、rag、processor等逻辑
        if self.agent_processor:
            question = self.agent_processor.preprocess(question)
        answer = self.agent_model.answer(question, **kwargs)
        if self.agent_processor:
            answer = self.agent_processor.postprocess(answer)
        return answer

    def stream_answer(self, question: str, **kwargs):
        # 流式输出答案
        if self.agent_processor:
            question = self.agent_processor.preprocess(question)
        for chunk in self.agent_model.stream_answer(question, **kwargs):
            if self.agent_processor:
                chunk = self.agent_processor.postprocess(chunk)
            yield chunk