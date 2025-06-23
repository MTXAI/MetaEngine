from smolagents import CodeAgent, ToolCallingAgent


class QaAgent(ToolCallingAgent):
    """
    A simple question-answering agent that uses a retriever tool to fetch relevant documents
    """

    def __init__(self, retriever_tool, model, history: str):
        super().__init__(
            tools=[retriever_tool],
            model=model,
            add_base_tools=True
        )
        self.history = history

    def run(self, task: str, stream: bool = False, reset: bool = True, images: list["PIL.Image.Image"] | None = None,
            additional_args: dict | None = None, max_steps: int | None = None):
        if self.history:
            task = f"{task}\n\nPrevious history:\n{self.history}"
        return super().run(task, stream, reset, images, additional_args, max_steps)



