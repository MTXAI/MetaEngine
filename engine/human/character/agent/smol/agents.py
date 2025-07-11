from smolagents import ToolCallingAgent


class QaAgent(ToolCallingAgent):
    """
    A simple question-answering agent that uses a retriever tool to fetch relevant documents
    """

    def __init__(self, retriever_tool, model, history: str):
        super().__init__(
            tools=[retriever_tool],
            model=model,
            add_base_tools=False
        )
        self.history = history

    def run(self, task: str, stream: bool = False, reset: bool = True, images: list["PIL.Image.Image"] | None = None,
            additional_args: dict | None = None, max_steps: int | None = None):
        if self.history:
            task = (f"Your task:{task}\n\nPrevious history:\n{self.history}\n\nPlease answer the question based on the previous history and the task")
        return super().run(task, stream, reset, images, additional_args, max_steps)



