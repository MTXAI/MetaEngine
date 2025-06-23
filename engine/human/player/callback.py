from engine.utils.async_utils import PipelineCallback


class FinalizerCallback(PipelineCallback):
    def __init__(self, client):
        self.client = client

    def on_stop(self):
        super().on_stop()
        self.client.finalizer()  # just an example
