import os

from engine.config import DEFAULT_RUNTIME_CONFIG
from engine.utils.concurrent.pool import ThreadPool

thread_pool = ThreadPool(
    max_workers=DEFAULT_RUNTIME_CONFIG.max_workers,
    max_queue_size=DEFAULT_RUNTIME_CONFIG.max_queue_size,
)

def set_environment_variables():
    # disable tokenizer fork
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

