from smolagents import CodeAgent


class SmolAgent(CodeAgent):
    """
    A custom agent class that extends CodeAgent.
    This class can be used to create agents with specific configurations or behaviors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional initialization can be done here if needed